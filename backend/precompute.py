

import pandas as pd
import numpy as np
import pickle
import os
import pymongo
import gridfs
from sklearn.feature_extraction.text import TfidfVectorizer


POSTS_PATH     = "posts.csv"
METADATA_PATH  = "influencer_master.csv"
SENTIMENT_PATH = "sentiment/all_influencers_sentiment.csv"
PKL_OUTPUT     = "tfidf_data.pkl"
MONGO_URI      = "mongodb+srv://user:password@cluster.mongodb.net/"
DB_NAME        = "AI_Influencer"
PKL_NAME       = "tfidf_data.pkl"

COLS_TO_KEEP = [
    "username", "full_name", "category", "country",
    "followers", "tier", "age_min", "age_max", "age_label",
    "avg_likes", "avg_comments", "engagement_rate",
    "fake_score", "is_fake", "sentiment_score", "total_comments",
    "positive_pct", "neutral_pct", "negative_pct",
    "caption", "row_id"
]

def classify_tier(followers):
    if followers < 10000:
        return "nano"
    elif followers < 100000:
        return "micro"
    elif followers < 500000:
        return "mid"
    elif followers < 1000000:
        return "macro"
    else:
        return "mega"


print("=" * 60)
print("  PRECOMPUTE — Influencer Marketing Platform")
print("=" * 60)

print("\n[1/6] Loading CSV files...")

posts     = pd.read_csv(POSTS_PATH,     encoding="utf-8")
metadata  = pd.read_csv(METADATA_PATH,  encoding="utf-8")
sentiment = pd.read_csv(SENTIMENT_PATH, encoding="utf-8")

print(f"   Posts loaded          : {len(posts)} rows")
print(f"   Influencers loaded    : {len(metadata)} rows")
print(f"   Sentiment loaded      : {len(sentiment)} rows")

posts["username"] = posts["username"].astype(str).str.strip().str.lower()
posts["caption"]  = posts["caption"].fillna("").astype(str).str.strip().str.lower()
posts["likes"]    = pd.to_numeric(posts["likes"], errors="coerce").fillna(0)

metadata["username"]        = metadata["username"].astype(str).str.strip().str.lower()
metadata["category"]        = metadata["category"].fillna("other").astype(str).str.strip().str.lower()
metadata["country"]         = metadata["country"].fillna("").astype(str).str.strip().str.lower()
metadata["followers"]       = pd.to_numeric(metadata["followers"],       errors="coerce").fillna(0)
metadata["age_min"]         = pd.to_numeric(metadata["age_min"],         errors="coerce").fillna(0)
metadata["age_max"]         = pd.to_numeric(metadata["age_max"],         errors="coerce").fillna(99)
metadata["engagement_rate"] = pd.to_numeric(metadata["engagement_rate"], errors="coerce").fillna(0)
metadata["fake_score"]      = pd.to_numeric(metadata["fake_score"],      errors="coerce").fillna(0)
metadata["is_fake"]         = metadata["is_fake"].astype(str).str.upper().str.strip() == "TRUE"
metadata["avg_likes"]       = pd.to_numeric(metadata["avg_likes"],       errors="coerce").fillna(0)
metadata["avg_comments"]    = pd.to_numeric(metadata["avg_comments"],    errors="coerce").fillna(0)


sentiment["username"] = sentiment["username"].astype(str).str.strip().str.lower()


print("\n[2/6] Computing tier and engagement rate...")

metadata["tier"] = metadata["followers"].apply(classify_tier)

metadata["engagement_rate"] = (
    (metadata["avg_likes"] + metadata["avg_comments"]) /
    metadata["followers"].replace(0, 1)
)

metadata.to_csv(METADATA_PATH, index=False, encoding="utf-8")
print(f"   Tier column saved to {METADATA_PATH}")

tier_counts = metadata["tier"].value_counts()
for tier, count in tier_counts.items():
    print(f"   {tier:<8}: {count} influencers")

print("\n[3/6] Aggregating captions and merging sentiment...")

posts["caption"] = posts["caption"].str[:500]

caption_agg = (
    posts.groupby("username")["caption"]
    .apply(lambda x: " ".join(x))
    .reset_index()
)

df = metadata.merge(caption_agg, on="username", how="left")
df["caption"] = df["caption"].fillna("")

df = df.merge(
    sentiment[["username", "sentiment_score", "total_comments",
               "positive_pct", "neutral_pct", "negative_pct"]],
    on="username", how="left"
)

df["sentiment_score"] = df["sentiment_score"].fillna(0.5)
df["total_comments"]  = df["total_comments"].fillna(0)
df["positive_pct"]    = df["positive_pct"].fillna(0)
df["neutral_pct"]     = df["neutral_pct"].fillna(0)
df["negative_pct"]    = df["negative_pct"].fillna(0)

df = df.reset_index(drop=True)
df["row_id"] = df.index

print(f"   Final influencer df   : {len(df)} rows")
print(f"   Captions merged       : {(df['caption'] != '').sum()} influencers have posts")
print(f"   Sentiment merged      : {(df['sentiment_score'] != 0.5).sum()} influencers have sentiment")


print("\n[4/6] Fitting TF-IDF model...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=500   
)

tfidf_matrix = vectorizer.fit_transform(df["caption"])

print(f"   Vocabulary size       : {len(vectorizer.vocabulary_)} terms")
print(f"   Matrix shape          : {tfidf_matrix.shape}")

df = df[[c for c in COLS_TO_KEEP if c in df.columns]]
print(f"   Columns kept          : {len(df.columns)} columns")

print(f"\n[5/6] Saving pkl locally...")
with open(PKL_OUTPUT, "wb") as f:
    pickle.dump({
        "vectorizer"   : vectorizer,
        "tfidf_matrix" : tfidf_matrix,
        "influencer_df": df
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

pkl_size = os.path.getsize(PKL_OUTPUT) / (1024 * 1024)
print(f"   Saved {PKL_OUTPUT} — {pkl_size:.1f} MB")

print(f"\n[6/6] Uploading pkl to MongoDB GridFS...")

try:
    client = pymongo.MongoClient(MONGO_URI)
    client.admin.command("ping")
    print("   Connected to MongoDB Atlas ")

    db_mongo = client[DB_NAME]
    fs       = gridfs.GridFS(db_mongo)

    existing = fs.find({"filename": PKL_NAME})
    deleted  = 0
    for old_file in existing:
        fs.delete(old_file._id)
        deleted += 1
    if deleted > 0:
        print(f"   Deleted {deleted} old version(s) from GridFS")

    with open(PKL_OUTPUT, "rb") as f:
        file_id = fs.put(f, filename=PKL_NAME)

    print(f"   Uploaded successfully ")
    print(f"   GridFS file ID : {file_id}")
    print(f"   File size      : {pkl_size:.1f} MB")

    # verify
    stored = fs.find_one({"filename": PKL_NAME})
    if stored:
        print(f"   Verified in GridFS — uploaded {stored.upload_date}")
    else:
        print(f"   Verification failed ")

    client.close()

except Exception as e:
    print(f"   MongoDB upload failed  — {e}")
    print(f"   pkl saved locally at {PKL_OUTPUT} — upload manually if needed")


print("  PRECOMPUTE COMPLETE")
print(f"  {METADATA_PATH:<30} — updated with tier column")
print(f"  {PKL_OUTPUT:<30} — {pkl_size:.1f} MB — saved + uploaded to GridFS")
print(f"\n  Flask API will download pkl from GridFS at startup.")
print(f"  Do NOT rerun unless your CSV data changes.")
