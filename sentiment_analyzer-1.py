# ================================================
# Social Media Sentiment Analyzer
# Built by: Midhat Tariq
# Description: Analyzes social media comments and
# classifies them as Positive, Negative, or Neutral
# ================================================

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# ── 1. Load comments from CSV ──────────────────
df = pd.read_csv("sample_comments.csv")
print(f"Loaded {len(df)} comments.\n")

# ── 2. Analyze sentiment ───────────────────────
def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Comment"].apply(analyze_sentiment)

# ── 3. Print results ───────────────────────────
print("===== SENTIMENT RESULTS =====")
for _, row in df.iterrows():
    print(f"[{row['Sentiment']}] {row['Comment']}")

# ── 4. Summary stats ───────────────────────────
print("\n===== SUMMARY =====")
summary = df["Sentiment"].value_counts()
print(summary)

# ── 5. Save results to CSV ─────────────────────
df.to_csv("results.csv", index=False)
print("\nResults saved to results.csv")

# ── 6. Plot pie chart ──────────────────────────
colors = ["#22c55e", "#ef4444", "#f59e0b"]
summary.plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=colors,
    startangle=140,
    title="Sentiment Analysis Results"
)
plt.ylabel("")
plt.tight_layout()
plt.savefig("sentiment_chart.png")
plt.show()
print("Chart saved as sentiment_chart.png")
