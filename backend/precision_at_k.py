"""
precision_at_k.py
═══════════════════════════════════════════════════════════════
Evaluates TF-IDF ranking quality using Precision@K.

Definition:
    P@K = number of relevant influencers in top K / K

Relevance Definition (proxy ground truth — no manual labels):
    An influencer is "relevant" if their relevance_score_norm > 0.1
    This means TF-IDF found a genuine caption match beyond just
    passing the category filter.

Test Campaigns:
    3-5 campaigns across different categories and locations.
    Each campaign is evaluated at K = 5, 10, 20.

Run:
    python precision_at_k.py
═══════════════════════════════════════════════════════════════
"""

from run_tfidf import InfluencerRanker
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── RELEVANCE THRESHOLD ──────────────────────────────────────
# influencer is "relevant" if TF-IDF score is above this value
RELEVANCE_THRESHOLD = 0.1

# ── K VALUES TO EVALUATE ─────────────────────────────────────
K_VALUES = [5, 10, 20]

# ── TEST CAMPAIGNS ───────────────────────────────────────────
TEST_CAMPAIGNS = [
    {
        "campaign_name": "Dove Shampoo",
        "description"  : "a shampoo dedicated for anti frizz and shiny hair care",
        "category"     : "beauty",
        "location"     : "usa",
        "age_min"      : 18,
        "age_max"      : 35,
        "budget"       : None,
    },
    {
        "campaign_name": "Pet4Life",
        "description"  : "healthy food and care products for your beloved pets",
        "category"     : "pet",
        "location"     : "india",
        "age_min"      : 18,
        "age_max"      : 40,
        "budget"       : None,
    },
    {
        "campaign_name": "Indigo Clothes",
        "description"  : "trendy fashion clothes for everyone find the right fit",
        "category"     : "fashion",
        "location"     : "uk",
        "age_min"      : 20,
        "age_max"      : 40,
        "budget"       : 4000,
    },
    {
        "campaign_name": "FitLife Gym",
        "description"  : "workout routines fitness tips and gym gear for active people",
        "category"     : "fitness",
        "location"     : "australia",
        "age_min"      : 18,
        "age_max"      : 40,
        "budget"       : None,
    },
    {
        "campaign_name": "Tasty Bites",
        "description"  : "delicious recipes and food reviews for home cooking lovers",
        "category"     : "food",
        "location"     : "canada",
        "age_min"      : 20,
        "age_max"      : 45,
        "budget"       : None,
    },
]

# ── PRECISION@K FUNCTION ─────────────────────────────────────
def precision_at_k(ranked_df, k, threshold=RELEVANCE_THRESHOLD):
    top_k    = ranked_df.head(k)
    relevant = (top_k["relevance_score_norm"] > threshold).sum()
    p_at_k   = relevant / k
    return round(p_at_k, 4), relevant


# ── MAIN ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  PRECISION@K EVALUATION — TF-IDF RANKING")
    print("=" * 65)
    print(f"  Relevance threshold : relevance_score_norm > {RELEVANCE_THRESHOLD}")
    print(f"  K values evaluated  : {K_VALUES}")
    print(f"  Test campaigns      : {len(TEST_CAMPAIGNS)}")
    print("=" * 65)

    # load from pkl — skips CSV loading and TF-IDF fitting
    import pickle
    ranker = InfluencerRanker()
    print("\n Loading precomputed data...")
    with open("tfidf_data.pkl", "rb") as f:
        data = pickle.load(f)
    ranker.vectorizer    = data["vectorizer"]
    ranker.tfidf_matrix  = data["tfidf_matrix"]
    ranker.influencer_df = data["influencer_df"]
    print(" Ready.\n")

    all_results = []

    for i, campaign in enumerate(TEST_CAMPAIGNS):
        print(f"{'─' * 65}")
        print(f"  Campaign {i+1}: {campaign['campaign_name']}")
        print(f"  Category : {campaign['category']}  |  "
              f"Location : {campaign['location']}  |  "
              f"Age : {campaign['age_min']}-{campaign['age_max']}")
        print(f"{'─' * 65}")

        try:
            ranked = ranker.rank_influencers(
                campaign_name = campaign["campaign_name"],
                description   = campaign["description"],
                category      = campaign["category"],
                location      = campaign["location"],
                age_min       = campaign["age_min"],
                age_max       = campaign["age_max"],
                budget        = campaign["budget"],
                top_n         = 20
            )

            total_returned = len(ranked)
            print(f"\n  Total influencers returned : {total_returned}\n")

            row = {"Campaign": campaign["campaign_name"],
                   "Category": campaign["category"]}

            for k in K_VALUES:
                # ── header for this K ─────────────────────────
                print(f"  {'═' * 55}")
                print(f"  Evaluating K = {k}")
                print(f"  {'═' * 55}")

                # if fewer results than K, skip
                if total_returned < k:
                    print(f"  Skipped — not enough results ({total_returned} < {k})\n")
                    row[f"P@{k}"] = "N/A"
                    continue

                p, relevant = precision_at_k(ranked, k)

                # assessment label
                if p >= 0.8:
                    assessment = "Excellent"
                elif p >= 0.6:
                    assessment = "Good"
                elif p >= 0.4:
                    assessment = "Fair"
                else:
                    assessment = "Poor"

                # display each influencer in top K
                print(f"  {'Rank':<6} {'Username':<30} {'Relevance':>10} {'Final':>8}  Status")
                print(f"  {'─' * 63}")
                top_k_df = ranked.head(k)
                for _, inf_row in top_k_df.iterrows():
                    is_relevant = inf_row["relevance_score_norm"] > RELEVANCE_THRESHOLD
                    tag = "✔ relevant" if is_relevant else "✘ not relevant"
                    print(f"  {int(inf_row['rank']):<6} "
                          f"{str(inf_row['username']):<30} "
                          f"{inf_row['relevance_score_norm']:>10.4f} "
                          f"{inf_row['final_relevance_score']:>8.4f}  {tag}")

                print(f"\n  P@{k} = {relevant}/{k} relevant = {p}  → {assessment}\n")

                row[f"P@{k}"] = p

            all_results.append(row)

        except ValueError as e:
            print(f"  Skipped — {e}\n")
            all_results.append({
                "Campaign": campaign["campaign_name"],
                "Category": campaign["category"],
                "P@5": "SKIP", "P@10": "SKIP", "P@20": "SKIP"
            })

    # ── SUMMARY TABLE ────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  FINAL PRECISION@K SUMMARY")
    print(f"{'=' * 65}")
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    # ── AVERAGE P@K ───────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  AVERAGE SCORES ACROSS ALL CAMPAIGNS:")
    for k in K_VALUES:
        col = f"P@{k}"
        if col in summary_df.columns:
            numeric = pd.to_numeric(summary_df[col], errors="coerce").dropna()
            if len(numeric) > 0:
                print(f"  Average P@{k:<4}: {numeric.mean():.4f}  "
                      f"({numeric.mean()*100:.1f}%)")

    print(f"\n{'=' * 65}")
    print("  NOTE: Relevance is defined as relevance_score_norm > 0.1")
    print("  This is a proxy ground truth based on TF-IDF caption match.")
    print("  For thesis/report: cite this as 'proxy-label evaluation'")
    print("  since no human-annotated ground truth labels are available.")
    print(f"{'=' * 65}")

    # ── PLOT ─────────────────────────────────────────────────
    print("\n  Generating Precision@K chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Precision@K Evaluation — TF-IDF Influencer Ranking",
        fontsize=14, fontweight="bold"
    )

    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]

    # ── LEFT: Line chart — P@K per campaign ──────────────────
    ax1 = axes[0]
    for idx, row in summary_df.iterrows():
        y_vals = []
        x_vals = []
        for k in K_VALUES:
            val = row.get(f"P@{k}", None)
            if val not in [None, "N/A", "SKIP"]:
                x_vals.append(k)
                y_vals.append(float(val))
        if x_vals:
            ax1.plot(
                x_vals, y_vals,
                marker="o", linewidth=2, markersize=8,
                label=row["Campaign"],
                color=colors[idx % len(colors)]
            )
            for x, y in zip(x_vals, y_vals):
                ax1.annotate(f"{y:.2f}", (x, y),
                             textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=8)

    ax1.set_title("Precision@K per Campaign", fontweight="bold")
    ax1.set_xlabel("K (Number of Top Results)")
    ax1.set_ylabel("Precision Score (0–1)")
    ax1.set_xticks(K_VALUES)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(0.8, color="green",  linestyle="--", alpha=0.4, label="0.8 (Excellent)")
    ax1.axhline(0.6, color="orange", linestyle="--", alpha=0.4, label="0.6 (Good)")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # ── RIGHT: Bar chart — Average P@K ───────────────────────
    ax2 = axes[1]
    avg_scores = []
    valid_k    = []
    for k in K_VALUES:
        col     = f"P@{k}"
        numeric = pd.to_numeric(summary_df[col], errors="coerce").dropna()
        if len(numeric) > 0:
            avg_scores.append(numeric.mean())
            valid_k.append(f"P@{k}")

    bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax2.bar(valid_k, avg_scores,
                   color=bar_colors[:len(valid_k)], alpha=0.85, width=0.4)

    ax2.set_title("Average Precision@K Across All Campaigns", fontweight="bold")
    ax2.set_xlabel("K Value")
    ax2.set_ylabel("Average Precision Score (0–1)")
    ax2.set_ylim(0, 1.15)
    ax2.axhline(0.8, color="green",  linestyle="--", alpha=0.4, label="0.8 Excellent")
    ax2.axhline(0.6, color="orange", linestyle="--", alpha=0.4, label="0.6 Good")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, avg_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.02, f"{val:.4f}",
                 ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("precision_at_k_chart.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Chart saved: precision_at_k_chart.png")


if __name__ == "__main__":
    main()