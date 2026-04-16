import json
import pandas as pd
import numpy as np
import sys

MASTER_FILE  = "user_final.json"     
POSTS_FILE   = "posts.csv"
COMMENTS_FILE = "comments.csv"
OUTPUT_JSON  = "this_is_final.json"
OUTPUT_CSV   = "influencer_master_final.csv"

FAKE_THRESHOLD = 0.6                     

W_ENGAGEMENT  = 0.35 
W_FOLLOW_RATIO = 0.25 
W_POST_ENGAGE  = 0.25 
W_COMMENT_LIKE = 0.15  

def load_data():
    print(" Loading master file...")
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)
    print(f"   {len(users)} users loaded")

    #load the csv file and then then convert the data to string, also converts to lower case.
    print(" Loading posts.csv...")
    posts = pd.read_csv(POSTS_FILE, encoding="utf-8")
    posts["username"] = posts["username"].astype(str).str.strip().str.lower()
    posts["likes"]    = pd.to_numeric(posts["likes"], errors="coerce").fillna(0)
    print(f"    {len(posts)} posts loaded")
#strip removes the unnecessary characters. for example we can remove the tab spaces, newlines. 
    print(" Loading comments.csv...")
    comments = pd.read_csv(COMMENTS_FILE, encoding="utf-8")
    comments["post_id"] = comments["post_id"].astype(str).str.strip()
    posts["post_id"]    = posts["post_id"].astype(str).str.strip()
    print(f"   {len(comments)} comments loaded")

    return users, posts, comments

def compute_post_stats(posts, comments):
    print("\n  Computing post statistics...")

    post_stats = (
        posts.groupby("username")
        .agg(
            avg_likes    = ("likes", "mean"),
            total_posts_actual = ("post_id", "count")
        )
        .reset_index()
    )

    comments_per_post = (
        comments.groupby("post_id")
        .size()
        .reset_index(name="comment_count")
    )

    posts_with_comments = posts.merge(
        comments_per_post,
        on="post_id",
        how="left"
    )
    posts_with_comments["comment_count"] = (
        posts_with_comments["comment_count"].fillna(0)
    )

    comment_stats = (
        posts_with_comments.groupby("username")
        .agg(avg_comments=("comment_count", "mean"))
        .reset_index()
    )

    stats = post_stats.merge(comment_stats, on="username", how="left")
    stats["avg_comments"] = stats["avg_comments"].fillna(0)

    print(f"    Stats computed for {len(stats)} users")
    return stats

def compute_signals(user: dict, stats_row):
    followers  = max(int(user.get("followers") or 0), 1)
    followees  = max(int(user.get("followees") or user.get("following") or 0), 0)
    total_posts = max(int(user.get("total_posts") or 0), 1)


    if stats_row is not None:
        avg_likes    = float(stats_row.get("avg_likes", 0))
        avg_comments = float(stats_row.get("avg_comments", 0))
        actual_posts = float(stats_row.get("total_posts_actual", total_posts))
    else:
        avg_likes    = 0.0
        avg_comments = 0.0
        actual_posts = float(total_posts)

 
    engagement_rate = (avg_likes + avg_comments) / followers

    if engagement_rate < 0.001:          
        sig_engagement = 1.0
    elif engagement_rate > 0.5:         
        sig_engagement = 0.8
    elif engagement_rate < 0.01:        
        sig_engagement = 0.5
    else:
        sig_engagement = 0.0             


    follow_ratio = followees / followers

    if follow_ratio > 2.0:              
        sig_follow = 1.0
    elif follow_ratio > 1.0:
        sig_follow = 0.6
    elif follow_ratio > 0.5:
        sig_follow = 0.2
    else:
        sig_follow = 0.0

    likes_per_post = avg_likes
    posts_engagement_ratio = likes_per_post / actual_posts

    if posts_engagement_ratio < 1.0 and actual_posts > 50:
        sig_post = 1.0
    elif posts_engagement_ratio < 5.0 and actual_posts > 20:
        sig_post = 0.5
    else:
        sig_post = 0.0


    if avg_likes > 0:
        comment_like_ratio = avg_comments / avg_likes
    else:
        comment_like_ratio = 0.0

    if avg_likes > 100 and comment_like_ratio < 0.005:
        sig_comment = 0.8                
    elif comment_like_ratio < 0.001:
        sig_comment = 0.5
    else:
        sig_comment = 0.0

    fake_score = (
        W_ENGAGEMENT   * sig_engagement +
        W_FOLLOW_RATIO * sig_follow     +
        W_POST_ENGAGE  * sig_post       +
        W_COMMENT_LIKE * sig_comment
    )

    return {
        "engagement_rate":    round(engagement_rate, 6),
        "following_ratio":    round(follow_ratio, 4),
        "comment_like_ratio": round(comment_like_ratio, 4),
        "avg_likes":          round(avg_likes, 2),
        "avg_comments":       round(avg_comments, 2),
        "fake_score":         round(fake_score, 4),
        "is_fake":            fake_score >= FAKE_THRESHOLD
    }

def main():
    print(" Fake Follower Detection\n")

    users, posts, comments = load_data()
    stats_df = compute_post_stats(posts, comments)

   
    stats_index = stats_df.set_index("username").to_dict(orient="index")

    print(f"\n  Computing fake scores for {len(users)} users...")

    fake_count  = 0
    valid_count = 0

    for i, user in enumerate(users):
        uname = str(user.get("username", "")).strip().lower()
        stats_row = stats_index.get(uname, None)

        signals = compute_signals(user, stats_row)
        user.update(signals)

        if user["is_fake"]:
            fake_count += 1
        else:
            valid_count += 1

        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1}/{len(users)}...")

    print(f"\nResults:")
    print(f"    Valid influencers : {valid_count}")
    print(f"    Flagged as fake   : {fake_count}")
    print(f"    Fake rate         : {round(fake_count/len(users)*100, 2)}%")

    #  Save JSON 
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    print(f"\n Saved JSON → {OUTPUT_JSON}")

    # Save CSV 
    df = pd.DataFrame(users)

    # flatten age_range dict if present
    if "age_range" in df.columns:
        df = df.drop(columns=["age_range"], errors="ignore")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved CSV  → {OUTPUT_CSV}")

    # Score distribution
    scores = [u["fake_score"] for u in users]
    print(f"\n Fake score distribution:")
    print(f"   Min   : {round(min(scores), 4)}")
    print(f"   Max   : {round(max(scores), 4)}")
    print(f"   Mean  : {round(np.mean(scores), 4)}")
    print(f"   Median: {round(np.median(scores), 4)}")

if __name__ == "__main__":
    main()