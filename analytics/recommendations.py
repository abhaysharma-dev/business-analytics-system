import pandas as pd
def gen_recos(df_: pd.DataFrame):
        recos = []
        total = len(df_)
        negc = (df_["sentiment"] == "negative").sum()
        if total > 0 and negc/total > 0.4:
            recos.append("Overall negative sentiment is high (>40%). Consider immediate coaching for counselors and revising scripts.")

        if "location" in df_.columns:
            by_loc = df_.groupby("location")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for loc, ratio in by_loc.items():
                if pd.notna(loc) and ratio >= 0.35:
                    recos.append(f"{loc}: Negative ratio {ratio:.0%}. Trial: reduce fees, add evening/weekend batches, or senior counselor callbacks.")

        if "tech_stack" in df_.columns:
            by_stack = df_.groupby("tech_stack")["sentiment"].apply(lambda s: (s=="negative").mean()).sort_values(ascending=False)
            for stack, ratio in by_stack.items():
                if pd.notna(stack) and ratio >= 0.35:
                    tips = {
                        "python": "emphasize job outcomes with case studies, add mini capstone demo",
                        "java": "offer installment plans, highlight placement partners",
                        "mern": "show live project repos and alumni testimonials",
                        "ai": "clarify math prerequisites and provide bridge modules"
                    }
                    extra = ""
                    k = str(stack).lower()
                    for key, val in tips.items():
                        if key in k:
                            extra = "; " + val
                            break
                    recos.append(f"{stack}: Negative ratio {ratio:.0%}. Address objections via FAQs{extra}.")

        text_all = " ".join(df_.get("combined_text", pd.Series(dtype=str)).dropna().astype(str).tolist()).lower()
        if any(k in text_all for k in ["fee", "fees", "price", "cost", "expensive"]):
            recos.append("Many fee-related objections → try scholarships, limited-time discounts, or EMI options.")
        if any(k in text_all for k in ["time", "timing", "slot", "schedule", "evening", "weekend"]):
            recos.append("Timing objections → add evening/weekend batches and flexible slots.")
        if any(k in text_all for k in ["location", "distance", "noida", "lucknow", "commute"]):
            recos.append("Location/commute issues → promote online/hybrid option and campus transfer flexibility.")
        if any(k in text_all for k in ["doubt", "support", "mentor", "teacher", "faculty"]):
            recos.append("Learning support concerns → advertise mentorship hours, doubt-solving sessions, and WhatsApp/Slack groups.")
        if any(k in text_all for k in ["job", "placement", "interview", "resume"]):
            recos.append("Career outcomes focus → showcase placement stats, resume/interview prep workshops.")
        return recos