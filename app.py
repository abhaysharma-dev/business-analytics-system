#----------BUSINESS ANALYTICS SYSTEM---------
import os
import io
import tempfile
from datetime import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from nlp.hf_model import load_hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import VECTORIZER_PATH, MODEL_PATH
from database.connection import (
    ensure_tables_exist,
    get_transcript_from_db,
    save_transcript_to_db,
    save_live_prediction_db,
    upsert_call_log,
    save_merged_sentiment_row,
    get_all_transcripts_from_db,
)
from nlp.preprocessing import spacy_lemmatizer_tokenizer
from models.evaluation import get_cv_report
from models.train import train_final_model
from models.predict import predict_sentiment
from analytics.recommendations import gen_recos
# Whisper
from asr.whisper_asr import load_whisper, transcribe_with_whisper
# HuggingFace
from transformers import pipeline
import joblib

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="Business Analytics System", layout="wide")
st.title("Business Analytics System")
st.caption("Audio → Transcripts → Sentiment → DB → Analytics")

# ===============================
# DB INIT
# ===============================
try:
    ensure_tables_exist()
except Exception as e:
    st.error(f"Database setup error: {e}")
    st.stop()

# ===============================
# MODEL LOAD / SAVE
# ===============================
@st.cache_resource(show_spinner=False)
def load_saved_model():
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        try:
            return joblib.load(VECTORIZER_PATH), joblib.load(MODEL_PATH)
        except:
            return None
    return None

def save_model(vectorizer, clf):
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)

saved = load_saved_model()
if saved:
    st.session_state["sentiment_model"] = saved
    st.session_state["model_trained"] = True
    st.sidebar.success("Saved sentiment model loaded from disk")
else:
    st.sidebar.warning("No saved model found. Train a model first.")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Settings")

whisper_size = st.sidebar.selectbox("Whisper model size", ["base", "small", "medium"], index=0)
use_hf = st.sidebar.checkbox("Force HuggingFace sentiment", value=False)
save_intermediate = st.sidebar.checkbox("Save processed CSV", value=True)

# ===============================
# UPLOAD SECTION
# ===============================
st.subheader("1) Upload Data")

col1, col2 = st.columns(2)
with col1:
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
with col2:
    audio_files = st.file_uploader(
        "Upload call recordings",
        type=["mp3", "wav", "m4a", "aac"],
        accept_multiple_files=True
    )

transcribe_only_new = st.checkbox("Transcribe only new calls", value=True)

# ===============================
# TRANSCRIPTION
# ===============================
whisper_model = None
if audio_files:
    whisper_model = load_whisper(whisper_size)

transcripts = []
if audio_files:
    st.subheader("2) Transcription")
    prog = st.progress(0)

    for i, f in enumerate(audio_files):
        cid = os.path.splitext(f.name)[0]
        existing = get_transcript_from_db(cid)

        if existing and transcribe_only_new:
            transcripts.append({"call_id": cid, "transcript_text": existing})
        else:
            audio_bytes = f.read()
            text = transcribe_with_whisper(audio_bytes, whisper_model, f.name)
            save_transcript_to_db(cid, text)
            transcripts.append({"call_id": cid, "transcript_text": text})

        prog.progress(int(((i + 1) / len(audio_files)) * 100))
    st.success(f"Processed {len(transcripts)} audio file(s)")
else:
    # keep empty DataFrame structure
    transcripts = []

# Convert to DataFrame
if transcripts:
    df_new = pd.DataFrame(transcripts).drop_duplicates(subset=["call_id"]).reset_index(drop=True)
else:
    df_new = pd.DataFrame(columns=["call_id", "transcript_text"])

# ===============================
# FETCH DB TRANSCRIPTS
# ===============================
db_df = get_all_transcripts_from_db()
df_tr = pd.concat([df_new, db_df], ignore_index=True).drop_duplicates("call_id")

if "live_rows" not in st.session_state:
    st.session_state["live_rows"] = []

# ===============================
# CSV MERGE
# ===============================

# Merge CSV + transcripts 
merged = None
if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file)
    except Exception:
        df_raw = pd.read_csv(csv_file, encoding="latin-1")
    st.success(f"CSV loaded with shape {df_raw.shape}")

    # column mapping UI
    with st.expander("Map columns (flexible)"):
        cols = ["<none>"] + list(df_raw.columns)
        map_student = st.selectbox("Student Name column", cols, index=cols.index("student_name") if "student_name" in df_raw.columns else 0)
        map_stack = st.selectbox("Tech Stack column", cols, index=cols.index("tech_stack") if "tech_stack" in df_raw.columns else 0)
        map_loc = st.selectbox("Location column", cols, index=cols.index("location") if "location" in df_raw.columns else 0)
        map_remarks = st.selectbox("Remarks/Notes column", cols, index=cols.index("remarks") if "remarks" in df_raw.columns else 0)
        map_callid = st.selectbox("Call ID column ", cols, index=cols.index("call_id") if "call_id" in df_raw.columns else 0)
        map_label = st.selectbox("Sentiment label column (optional: positive/neutral/negative)", cols, index=cols.index("sentiment_label") if "sentiment_label" in df_raw.columns else 0)

    def pick(colname):
        return None if colname == "<none>" else df_raw[colname]

    df = pd.DataFrame({
        "call_id": pick(map_callid) if map_callid != "<none>" else pd.Series([None]*len(df_raw)),
        "student_name": pick(map_student) if map_student != "<none>" else pd.Series([None]*len(df_raw)),
        "tech_stack": pick(map_stack) if map_stack != "<none>" else pd.Series([None]*len(df_raw)),
        "location": pick(map_loc) if map_loc != "<none>" else pd.Series([None]*len(df_raw)),
        "remarks": pick(map_remarks) if map_remarks != "<none>" else pd.Series([""]*len(df_raw)),
        "label": pick(map_label) if map_label != "<none>" else pd.Series([None]*len(df_raw)),
    })

    if "call_id" in df.columns:
        df["call_id"] = df["call_id"].astype(str).str.replace(r'\.0$', '', regex=True)
    # Ensure transcript df_tr fetched
    if not df_tr.empty:
        df_tr["call_id"] = df_tr["call_id"].astype(str)
        merged = pd.merge(df, df_tr, on="call_id", how="left")
    else:
        # If no audio uploaded, still we want to try fetching transcripts from DB for call_ids in CSV
        # fetch transcripts for all CSV call_ids
        csv_call_ids = df["call_id"].dropna().unique().tolist()
        rows = []
        for cid in csv_call_ids:
            t = get_transcript_from_db(cid)
            rows.append({"call_id": cid, "transcript_text": t if t is not None else ""})
        db_tr_df = pd.DataFrame(rows)
        merged = pd.merge(df, db_tr_df, on="call_id", how="left")
    
    if st.session_state["live_rows"]:
        merged = pd.concat(
            [merged, pd.DataFrame(st.session_state["live_rows"])],
            ignore_index=True
        )

    merged["remarks"] = merged.get("remarks", pd.Series([""] * len(merged))).fillna("")
    merged["transcript_text"] = merged.get("transcript_text", "").fillna("")
    merged["combined_text"] = (merged["remarks"].astype(str) + " " + merged["transcript_text"].astype(str)).str.strip()
    merged["cleaned_text"] = merged["combined_text"].apply(lambda x: " ".join(spacy_lemmatizer_tokenizer(str(x))))
    
    # Save merged rows into call_logs table
    for _, row in merged.iterrows():
        upsert_call_log({
            "call_id": row.get("call_id"),
            "student_name": row.get("student_name"),
            "tech_stack": row.get("tech_stack"),
            "location": row.get("location"),
            "remarks": row.get("remarks"),
            "transcript_text": row.get("transcript_text"),
            "combined_text": row.get("combined_text"),
            "cleaned_text": row.get("cleaned_text"),
            "label": row.get("label"),
        })

        # Show merged results (only rows with transcript or show all if none)
    df_display = merged[merged['transcript_text'].notna() & (merged['transcript_text'] != '')]

    # columns to show in UI
    def visible_cols(df):
        return [c for c in df.columns if c not in ["remarks", "combined_text"]]

    if df_display.empty:
        st.info(
            "No transcripts found for CSV call_ids (or audio not uploaded). "
            "Showing the merged CSV data including cleaned text."
        )
        st.dataframe(
            merged[visible_cols(merged)].head(50),
            width="stretch",
            hide_index=True,
        )
    else:
        st.write(
            f"Showing {len(df_display)} merged row(s) that have a transcript (including cleaned text):"
        )
        st.dataframe(
            df_display[visible_cols(df_display)].head(50),
            width="stretch",
            hide_index=True,
        )

# ===============================
# TRAINING & ACCURACY
# ===============================
can_train = (merged is not None) and ("label" in merged.columns) and merged["label"].notna().any()

# custom train model if target column present and calculate accuracy
if can_train:
    st.info("Labels found in CSV – you can evaluate model with cross-validation.")

    train_data = merged[merged["label"].notna()]

    st.write("### Step 1: Evaluate Model (Cross-Validation)")
    if st.sidebar.button("Run Cross-Validation"):
        cv_summary = get_cv_report(
            train_data["combined_text"],
            train_data["label"],
            cv=6,   
        )

        st.success("Cross-validation complete (5 folds).")
        st.caption("Scores are mean ± std across folds. Higher is better.")
        st.dataframe(
            cv_summary.style.format("{:.3f}"),
            width="stretch"

        )

    st.write("### Step 2: Train Final Model")
    if st.sidebar.button("Train Final Model for Prediction"):
        vectorizer, clf = train_final_model(
            train_data["combined_text"],
            train_data["label"]
        )
        st.session_state["sentiment_model"] = (vectorizer, clf)
        st.session_state["model_trained"] = True
        save_model(vectorizer, clf)
        st.success("✅ Final sentiment model trained & saved.")
else:
    st.warning("No valid label column found. Accuracy cannot be calculated.")

# ===============================
# SENTIMENT APPLY
# ===============================
if merged is not None and len(merged) > 0:

    st.subheader("Apply Sentiment & Analyze")

    # 1️⃣ Force HuggingFace
    if use_hf:
        st.warning("Using HuggingFace DistilBERT (forced).")
        with st.spinner("Running HuggingFace sentiment model…"):
            hf = load_hf_pipeline()
            preds, scores = [], []

            for txt in merged["combined_text"].fillna(""):
                try:
                    r = hf(txt[:4096])[0]
                    label = r["label"].lower()

                    if label == "positive":
                        preds.append("positive")
                    elif label == "negative":
                        preds.append("negative")
                    else:
                        preds.append("neutral")

                    scores.append(float(r.get("score", 0)))
                except:
                    preds.append("neutral")
                    scores.append(0.0)

        merged["sentiment"] = preds
        merged["sentiment_score"] = scores
        model_used = "huggingface"

    # 2️⃣ Custom TF-IDF model
    elif st.session_state.get("model_trained"):
        st.success("Using custom TF-IDF model.")
        vectorizer, clf = st.session_state["sentiment_model"]

        X = vectorizer.transform(merged["combined_text"])
        probas = clf.predict_proba(X)

        merged["sentiment"] = clf.predict(X)
        merged["sentiment_score"] = probas.max(axis=1)
        model_used = "custom_tf_idf"

    # 3️⃣ Fallback HF
    else:
        st.warning("No trained model → HuggingFace fallback.")
        hf = load_hf_pipeline()
        preds, scores = [], []

        for txt in merged["combined_text"].fillna(""):
            try:
                r = hf(txt[:4096])[0]
                label = r["label"].lower()

                if label == "positive":
                    preds.append("positive")
                elif label == "negative":
                    preds.append("negative")
                else:
                    preds.append("neutral")

                scores.append(float(r.get("score", 0)))
            except:
                preds.append("neutral")
                scores.append(0.0)

        merged["sentiment"] = preds
        merged["sentiment_score"] = scores
        model_used = "huggingface_fallback"

    st.success(f"Sentiment applied using: {model_used}")
    # -----------------------------
    # Save sentiment back to DB
    # -----------------------------
    for _, r in merged.iterrows():
        upsert_call_log({
            "call_id": r.get("call_id"),
            "student_name": r.get("student_name"),
            "tech_stack": r.get("tech_stack"),
            "location": r.get("location"),
            "remarks": r.get("remarks"),
            "transcript_text": r.get("transcript_text"),
            "combined_text": r.get("combined_text"),
            "cleaned_text": r.get("cleaned_text"),
            "label": r.get("label"),
            "sentiment": r.get("sentiment"),
            "sentiment_score": float(r.get("sentiment_score"))
                    if r.get("sentiment_score") is not None else None
        })
    for _, r in merged.iterrows():
        try:
            save_merged_sentiment_row(r)
        except Exception as e:
            st.warning(f"Could not save row for call_id {r.get('call_id')}: {e}")

    # analytics- charts, keywords
    st.markdown("### Analytics")
    colA, colB, colC = st.columns(3)
    with colA:
        fig = px.pie(merged, names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig, width="stretch")
    with colB:
    # -----------------------------
    # Sentiment Pie Charts by Location 
    # -----------------------------
        st.subheader("Sentiment Distribution by Location (Pie Charts)")

        if "location" in merged.columns and "sentiment" in merged.columns:
            merged["location"] = merged["location"].astype(str).str.strip().str.title()
            locations = merged["location"].dropna().unique()

            for loc in locations:
                df_loc = merged[merged["location"] == loc]

                pos = (df_loc["sentiment"] == "positive").sum()
                neg = (df_loc["sentiment"] == "negative").sum()
                total = pos + neg

                if total == 0:
                    continue

                pos_pct = (pos / total) * 100
                neg_pct = (neg / total) * 100

                st.write(f"### {loc} — Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}%")
                # Make the pie chart
                fig = px.pie(
                    names=["Positive", "Negative"],
                    values=[pos, neg],
                    hole=0.35,
                )
                fig.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig,width="stretch")
        else:
            st.info("No location or sentiment data available.")


    with colC:
        if "tech_stack" in merged.columns:
            fig3 = px.bar(merged.fillna({"tech_stack":"Unknown"}), x="tech_stack", color="sentiment", title="Sentiment by Tech Stack")
            st.plotly_chart(fig3, width="stretch")

    # Top Negative Keywords 
    st.markdown("### Top Negative Keywords")
    neg_text_series = merged[merged["sentiment"] == "negative"]["combined_text"].dropna()
    if len(neg_text_series) >= 3:
        try:
            vec_keywords = TfidfVectorizer(ngram_range=(1, 2),tokenizer=spacy_lemmatizer_tokenizer, max_features=50)
            X_keywords = vec_keywords.fit_transform(neg_text_series)
            sums = np.asarray(X_keywords.sum(axis=0)).ravel()
            vocab = np.array(vec_keywords.get_feature_names_out())
            kw_df = pd.DataFrame({"keyword": vocab, "score": sums}).sort_values("score", ascending=False).head(20)
            fig5 = px.bar(kw_df, x="keyword", y="score", title="Top Negative Keywords (TF-IDF)")
            st.plotly_chart(fig5,width="stretch")
        except Exception as e:
            st.error(f"Could not generate keywords: {e}")
    else:
        st.info("Not enough negative samples to extract keywords.")

if csv_file is not None and merged is not None and len(merged) > 0:
    st.session_state["csv_loaded"] = True
else:
    st.session_state["csv_loaded"] = False

# Live predictions (upload new recordings to predict)
if st.session_state.get("model_trained") and st.session_state.get("csv_loaded"):
    st.subheader(" Predict on Live (Unseen) Calls")
    st.success("Your custom model is trained! Upload new audio files here for live prediction.")

    live_audio_files = st.file_uploader("Upload new recordings for live prediction", type=["mp3","wav","m4a","aac"], accept_multiple_files=True, key="live_uploader")
    if live_audio_files:
            vectorizer, clf = st.session_state["sentiment_model"]
            try:
                whisper_model = load_whisper(whisper_size)
            except Exception as e:
                st.error(f"Error loading Whisper model: {e}")
                whisper_model = None

            results_rows = []

            if "live_cache" not in st.session_state:
                st.session_state["live_cache"] = {}  

            live_cache = st.session_state["live_cache"]

            for audio_file in live_audio_files:
                cid = os.path.splitext(os.path.basename(audio_file.name))[0]
                st.markdown("---")
                st.write(f"**File:** {audio_file.name}")

                try:
                    audio_bytes = audio_file.read()
                    audio_file.seek(0)

                    if cid not in live_cache:
                        with st.spinner("Transcribing..."):
                            raw_transcript = transcribe_with_whisper(audio_bytes, whisper_model, audio_file.name)
                        live_cache[cid] = raw_transcript
                    else:
                        raw_transcript = live_cache[cid]

                    st.write(f"**Transcript:** {raw_transcript}")
                    # ---------- 2) FORM: MANUAL INPUT + BUTTON  ----------
                    name_key = f"live_name_{cid}"
                    loc_key = f"live_loc_{cid}"
                    stack_key = f"live_stack_{cid}"

                    if name_key not in st.session_state:
                        st.session_state[name_key] = ""
                    if loc_key not in st.session_state:
                        st.session_state[loc_key] = ""
                    if stack_key not in st.session_state:
                        st.session_state[stack_key] = ""

                    with st.form(f"live_form_{cid}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            name_input = st.text_input(
                                "Student Name",
                                key=name_key,
                            )
                        with col2:
                            loc_input = st.text_input(
                                "Location",
                                key=loc_key,
                            )
                        with col3:
                            stack_input = st.text_input(
                                "Tech Stack / Course",
                                key=stack_key,
                            )

                        submitted = st.form_submit_button("Save & Predict")
                    if not submitted:
                        continue

                    # ---------- PREDICTION (HF OR CUSTOM) ----------
                    with st.spinner("Predicting..."):
                        if use_hf:
                            nlp_ob = load_hf_pipeline()
                            r = nlp_ob(raw_transcript[:4096])[0]
                            label = r["label"].lower()
                            if label == "positive":
                                prediction = "positive"
                            elif label == "negative":
                                prediction = "negative"
                            else:
                                prediction = "neutral"
                            score = float(r.get("score", 0))
                            model_used = "huggingface"
                        else:
                            prediction, score = predict_sentiment(vectorizer, clf, raw_transcript)
                            model_used = "custom_tf_idf"

                    # ---------- SAVE TO DB + UPDATE MERGED ----------
                    save_live_prediction_db(cid, raw_transcript, prediction, score)

                    final_name = st.session_state[name_key].strip() or None
                    final_loc = st.session_state[loc_key].strip() or None
                    final_stack = st.session_state[stack_key].strip() or None
                    row_for_log = {
                        "call_id": cid,
                        "student_name": final_name,
                        "tech_stack": final_stack,
                        "location": final_loc,
                        "remarks": None,
                        "transcript_text": raw_transcript,
                        "combined_text": raw_transcript,
                        "cleaned_text": " ".join(spacy_lemmatizer_tokenizer(raw_transcript)),
                        "label": None,
                        "sentiment": prediction,
                        "sentiment_score": float(score) if score is not None else None,
                    }
                    # store live row for this session's analytics
                    st.session_state["live_rows"].append(row_for_log)

                    try:
                        upsert_call_log(row_for_log)
                    except Exception as e:
                        st.warning(f"Could not upsert live call_log for {cid}: {e}")

                    results_rows.append({
                        "call_id": cid,
                        "transcript": raw_transcript,
                        "prediction": prediction,
                        "score": score,
                        "timestamp": datetime.now().isoformat(),
                    })

                    # ----------  SHOW RESULT ----------
                    if prediction == "positive":
                        st.success(f"**Predicted Sentiment: {prediction.upper()}** (Confidence: {score:.2f})")
                    elif prediction == "negative":
                        st.error(f"**Predicted Sentiment: {prediction.upper()}** (Confidence: {score:.2f})")
                    else:
                        st.info(f"**Predicted Sentiment: {prediction.upper()}** (Confidence: {score:.2f})")

                except Exception as e:
                    st.error(f"Failed to process {audio_file.name}: {e}")
            # Export live predictions as CSV for user
            if results_rows:
                live_df = pd.DataFrame(results_rows)
                csv_buf = io.StringIO()
                live_df.to_csv(csv_buf, index=False)
                st.sidebar.download_button("Download live predictions CSV", data=csv_buf.getvalue(),
                                   file_name=f"live_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")
     
    else:
        st.info("Train a model and click the button to predict live.")

# ===============================
# RECOMMENDATIONS
# ===============================
if merged is not None:
    st.subheader("Recommendations")
    recos = gen_recos(merged)
    if recos:
        for r in recos:
            st.markdown(f"- {r}")
    else:
        st.info("No strong recommendations yet")

# ===============================
# EXPORT
# ===============================
if merged is not None and save_intermediate:
    st.subheader("Export Processed CSV")
    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    st.sidebar.download_button(
        "Download CSV",
        data=buf.getvalue(),
        file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
