#only for console backend testing 

from run_tfidf import InfluencerRanker
import sys
import re
import pandas as pd

VALID_CATEGORIES = ["beauty", "fashion", "fitness", "food", "interior", "pet", "travel", "family", "other"]

def validate_campaign_name(name):
    if len(name) < 3:
        return False, "Campaign name must be at least 3 characters."
    if not re.match(r"^[a-zA-Z0-9\s\-&'.]+$", name):
        return False, "Campaign name can only contain letters, numbers and basic punctuation."
    if not any(c.isalpha() for c in name):
        return False, "Campaign name must contain at least one letter."
    return True, ""

def validate_description(desc):
    if len(desc) < 20:
        return False, "Description must be at least 20 characters."

    words = [w for w in desc.split() if w.isalpha()]
    if len(words) < 4:
        return False, "Description must contain at least 4 real words."

    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len > 12:
        return False, "Description does not look valid. Please enter a real campaign description."

    junk = re.sub(r"[a-zA-Z0-9\s\.\!\?\,\'\-]", "", desc)
    if len(junk) > 3:
        return False, "Description contains invalid characters."

    if len(set(desc.lower().split())) < 4:
        return False, "Description must contain at least 4 unique words."

    for word in words:
        if len(set(word)) < len(word) * 0.4:
            return False, "Description does not look valid. Please enter a real campaign description."

    return True, ""

def validate_category(cat):
    if cat.lower() not in VALID_CATEGORIES:
        return False, f"Invalid category. Choose from: {', '.join(VALID_CATEGORIES)}"
    return True, ""

def validate_location(loc):
    if loc == "":
        return False, "Location is required."
    if not re.match(r"^[a-zA-Z\s\-]+$", loc):
        return False, "Location can only contain letters."
    if len(loc) < 2:
        return False, "Location must be at least 2 characters."
    return True, ""

def validate_budget(budget_str):

    if budget_str.strip() == "":
        return True, None, ""
    try:
        budget = float(budget_str)
        if budget < 100:
            return False, None, "Budget must be at least $100."
        return True, budget, ""
    except ValueError:
        return False, None, "Budget must be a number."

def validate_age(age_str, label):
    # age is required — empty input is rejected
    if age_str.strip() == "":
        return False, None, f"{label} is required."
    try:
        age = int(age_str)
        if age < 13:
            return False, None, f"{label} must be at least 13."
        if age > 80:
            return False, None, f"{label} must be at most 80."
        return True, age, ""
    except ValueError:
        return False, None, f"{label} must be a whole number."

def validate_top_n(n_str):
    try:
        n = int(n_str)
        if n < 1:
            return False, None, "Must be at least 1."
        if n > 100:
            return False, None, "Must be at most 100."
        return True, n, ""
    except ValueError:
        return False, None, "Must be a whole number."


print("  Influencer Marketing Platform:")
print()

while True:
    campaign_name = input("Campaign Name           : ").strip()
    valid, msg = validate_campaign_name(campaign_name)
    if valid:
        break
    print(f"  {msg}")


while True:
    description = input("Campaign Description    : ").strip()
    valid, msg = validate_description(description)
    if valid:
        break
    print(f"  {msg}")

while True:
    print(f"  Valid categories: {', '.join(VALID_CATEGORIES)}")
    category = input("Category                : ").strip().lower()
    valid, msg = validate_category(category)
    if valid:
        break
    print(f"  {msg}")

while True:
    location = input("Location                : ").strip()
    valid, msg = validate_location(location)
    if valid:
        break
    print(f"  {msg}")

while True:
    budget_input = input("Budget ($ optional, press Enter to skip) : ").strip()
    valid, budget, msg = validate_budget(budget_input)
    if valid:
        break
    print(f"  {msg}")

if budget is None:
    print("  No budget limit — all tiers included.")

print("\n  Target Age Range (required)")

while True:
    age_min_input = input("   Min age             : ").strip()
    valid, age_min, msg = validate_age(age_min_input, "Min age")
    if valid:
        break
    print(f"   {msg}")

while True:
    age_max_input = input("   Max age             : ").strip()
    valid, age_max, msg = validate_age(age_max_input, "Max age")
    if valid:
        break
    print(f"   {msg}")

if age_min is None or age_max is None:
    print("  Both min and max age are required.")

# ensure min < max
while age_min >= age_max:
    print(f"  Min age ({age_min}) must be less than Max age ({age_max}). Re-enter both.")
    age_min_input = input("   Min age             : ").strip()
    _, age_min, _ = validate_age(age_min_input, "Min age")
    age_max_input = input("   Max age             : ").strip()
    _, age_max, _ = validate_age(age_max_input, "Max age")

while True:
    top_n_input = input("\nHow many top influencers to return? : ").strip()
    valid, top_n, msg = validate_top_n(top_n_input)
    if valid:
        break
    print(f"  {msg}")

try:
    ranker = InfluencerRanker()

    print("\n Loading precomputed data...")
    import pickle
    with open("tfidf_data.pkl", "rb") as f:
        data = pickle.load(f)

    ranker.vectorizer    = data["vectorizer"]
    ranker.tfidf_matrix  = data["tfidf_matrix"]
    ranker.influencer_df = data["influencer_df"]
    print(" Ready.")

    print(" Ranking influencers...")
    print()

    results = ranker.rank_influencers(
        campaign_name = campaign_name,
        description   = description,
        category      = category,
        location      = location,
        age_min       = age_min,
        age_max       = age_max,
        budget        = budget,
        top_n         = top_n
    )

    if len(results) < top_n:
        print(f"  Note: Only {len(results)} influencers found matching all filters.")

    print(f"\n Top {top_n} Influencers for '{campaign_name}':\n")
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.width", 200)
    print(results.to_string(index=False))

 

except ValueError as e:
    print(f"\n {e}")
    sys.exit()
except Exception as e:
    print(f"\n Error: {e}")
    sys.exit()