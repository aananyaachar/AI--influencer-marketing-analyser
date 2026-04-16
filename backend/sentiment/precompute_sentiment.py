import pandas as pd

print("Loading comments...")
comments = pd.read_csv("FINAL_VADER_BERT_REAL_SCORES.csv")
print(f"  Total comments : {len(comments):,}")
print(f"  Total unique influencers : {comments['username'].nunique():,}")

comments["weighted_score"] = (
    (0.7 * comments["bert_score"]) +
    (0.3 * comments["vader_score"])
)

print("\nCalculating sentiment per influencer...")
sentiment = comments.groupby("username").agg(
    total_comments  = ("translated_comment", "count"),
    sentiment_score = ("weighted_score",     "mean"),
    positive_count  = ("bert_sentiment",
                       lambda x: (x == "Positive").sum()),
    neutral_count   = ("bert_sentiment",
                       lambda x: (x == "Neutral").sum()),
    negative_count  = ("bert_sentiment",
                       lambda x: (x == "Negative").sum()),
    avg_bert_score  = ("bert_score",  "mean"),
    avg_vader_score = ("vader_score", "mean"),
).reset_index()


sentiment = sentiment.rename(
    columns={"username": "username"}
)

sentiment["positive_pct"] = (
    sentiment["positive_count"] /
    sentiment["total_comments"] * 100
).round(1)

sentiment["neutral_pct"] = (
    sentiment["neutral_count"] /
    sentiment["total_comments"] * 100
).round(1)

sentiment["negative_pct"] = (
    sentiment["negative_count"] /
    sentiment["total_comments"] * 100
).round(1)

sentiment["sentiment_score"]  = sentiment["sentiment_score"].round(4)
sentiment["avg_bert_score"]   = sentiment["avg_bert_score"].round(4)
sentiment["avg_vader_score"]  = sentiment["avg_vader_score"].round(4)

sentiment = sentiment.sort_values(
    "sentiment_score", ascending=False
).reset_index(drop=True)

sentiment.to_csv("all_influencers_sentiment.csv", index=False)

print(f"  DONE — Pre-computation Complete")
print(f"  Total comments analyzed  : {len(comments):,}")
print(f"  Total influencers saved  : {len(sentiment):,}")
print(f"  Avg sentiment score      : {sentiment['sentiment_score'].mean():.4f}")
print(f"  Highest score            : {sentiment['sentiment_score'].max():.4f}"
      f"  (@{sentiment.iloc[0]['username']})")
print(f"  Lowest score             : {sentiment['sentiment_score'].min():.4f}"
      f"  (@{sentiment.iloc[-1]['username']})")
print(f"\n  File saved : all_influencers_sentiment.csv")
