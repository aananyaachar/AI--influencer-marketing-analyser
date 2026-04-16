import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InfluencerRanker:

    VALID_CATEGORIES = [
        "beauty", "fashion", "fitness",
        "food", "interior", "pet",
        "travel", "family", "other"
    ]

    BUDGET_TIER_MAP = [
        (1000,   ["nano"]),
        (10000,  ["nano", "micro"]),
        (50000,  ["nano", "micro", "mid"]),
        (100000, ["nano", "micro", "mid", "macro"]),
        (float("inf"), ["nano", "micro", "mid", "macro", "mega"]),
    ]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500        # matches precompute.py
        )
        self.influencer_df = None
        self.tfidf_matrix  = None

    @staticmethod
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

    def get_allowed_tiers(self, budget):
        if budget is None:
            return ["nano", "micro", "mid", "macro", "mega"]
        for threshold, tiers in self.BUDGET_TIER_MAP:
            if budget <= threshold:
                return tiers
        return ["nano", "micro", "mid", "macro", "mega"]

    def load_data(self, posts_path, metadata_path, sentiment_path):
        posts    = pd.read_csv(posts_path,    encoding="utf-8")
        metadata = pd.read_csv(metadata_path, encoding="utf-8")

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

        caption_agg = (
            posts.groupby("username")["caption"]
            .apply(lambda x: " ".join(x))
            .reset_index()
        )

        df = metadata.merge(caption_agg, on="username", how="left")
        df["caption"] = df["caption"].fillna("")
        df["tier"]    = df["followers"].apply(self.classify_tier)

        df["engagement_rate"] = (
            (df["avg_likes"] + df["avg_comments"]) /
            df["followers"].replace(0, 1)
        )

        df = df.reset_index(drop=True)
        df["row_id"] = df.index

        sentiment = pd.read_csv(sentiment_path, encoding="utf-8")
        sentiment["username"] = sentiment["username"].astype(str).str.strip().str.lower()

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

        self.influencer_df = df
        print(f"Loaded {len(df)} influencers")

    def fit(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.influencer_df["caption"]
        )
        print("TF-IDF model fitted......")

    def rank_influencers(
        self,
        campaign_name,
        description,
        category,
        location,
        age_min,
        age_max,
        budget=None,
        top_n=10
    ):
        if category.lower() not in [c.lower() for c in self.VALID_CATEGORIES]:
            raise ValueError(
                f"Category '{category}' not found in dataset. "
                f"Valid categories: {self.VALID_CATEGORIES}"
            )

        df = self.influencer_df.copy()

        df["fake_score"]      = pd.to_numeric(df["fake_score"],      errors="coerce").fillna(0)
        df["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce").fillna(0)
        df["age_min"]         = pd.to_numeric(df["age_min"],         errors="coerce").fillna(0)
        df["age_max"]         = pd.to_numeric(df["age_max"],         errors="coerce").fillna(99)
        df["followers"]       = pd.to_numeric(df["followers"],       errors="coerce").fillna(0)
        df["avg_likes"]       = pd.to_numeric(df["avg_likes"],       errors="coerce").fillna(0)
        df["avg_comments"]    = pd.to_numeric(df["avg_comments"],    errors="coerce").fillna(0)
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.5)
        df["is_fake"]         = df["is_fake"].astype(str).str.upper().str.strip() == "TRUE"
       

        df = df[df["fake_score"] <= 0.6]
        df = df[df["is_fake"] == False]
        print(f"   After fake pre-filter  : {len(df)} influencers")

        allowed_tiers = self.get_allowed_tiers(budget)
        df = df[df["tier"].isin(allowed_tiers)]
        print(f"   After budget filter    : {len(df)} influencers")

        df = df[df["engagement_rate"] >= 0.01]
        print(f"   After engagement filter: {len(df)} influencers")

        df = df[df["category"].str.lower() == category.lower()]
        print(f"   After category filter  : {len(df)} influencers")

        loc = location.strip().lower()
        df = df[df["country"].str.lower().str.split().apply(lambda words: loc in words)]
        print(f"   After location filter  : {len(df)} influencers")

        df = df[
            (df["age_min"] <= age_max) &
            (df["age_max"] >= age_min)
        ]
        print(f"   After age filter       : {len(df)} influencers")

        if df.empty:
            raise ValueError("No influencers match the given filters.")

        query     = f"{description} {category}"
        query_vec = self.vectorizer.transform([query])

        filtered_ids      = df["row_id"].values
        similarity_scores = cosine_similarity(
            query_vec,
            self.tfidf_matrix[filtered_ids]
        )[0]

        df = df.copy()
        df["relevance_score"] = similarity_scores

        df["engagement_score"] = (
            df["avg_likes"] + df["avg_comments"]
        ) / df["followers"].replace(0, 1)

        def normalise(series):
            mn, mx = series.min(), series.max()
            if mx == mn:
                return series * 0
            return (series - mn) / (mx - mn)

        df["relevance_score_norm"]  = normalise(df["relevance_score"])
        df["engagement_score_norm"] = normalise(df["engagement_score"])

        overlap = (
            df["age_max"].clip(upper=age_max) -
            df["age_min"].clip(lower=age_min)
        ).clip(lower=0)
        brand_range = max(age_max - age_min, 1)
        df["demography_score"] = (overlap / brand_range).clip(0, 1)

        df["final_relevance_score"] = (
            0.50 * df["relevance_score_norm"] +
            0.30 * df["engagement_score_norm"] +
            0.20 * df["demography_score"]
        )

        df["final_score"] = (
            0.70 * df["final_relevance_score"] +
            0.30 * df["sentiment_score"]
        )
        print("Ranking algorithm executed")
        ranked         = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        ranked["full_name"] = ranked["full_name"].astype(str).str[:20]

        return ranked.head(top_n)[[
            "rank", "username", "full_name", "tier", "category",
            "country", "followers", "age_label",
            "relevance_score_norm", "engagement_score_norm",
            "demography_score", "final_relevance_score",
            "sentiment_score", "positive_pct", "neutral_pct", "negative_pct",
            "final_score"
        ]]