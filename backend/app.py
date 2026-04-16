from bson import ObjectId
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from datetime import datetime
import pickle
import gridfs
import pandas as pd
import numpy as np
import re
import os

from run_tfidf import InfluencerRanker

app = Flask(__name__)
CORS(app)

MONGO_URI      = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://user:password@cluster.mongodb.net/"
)
DB_NAME        = "AI_Influencer"
PKL_LOCAL_PATH = "tfidf_data.pkl"

VALID_CATEGORIES = [
    "beauty", "fashion", "fitness", "food",
    "interior", "pet", "travel", "family", "other"
]


_rank_cache = {}

try:
    client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db         = client[DB_NAME]
    collection = db["campaign_result"]
    fs         = gridfs.GridFS(db)

    # create indexes for faster queries
    collection.create_index("results.username")
    collection.create_index([("created_at", DESCENDING)])

    print(" MongoDB connected successfully")
except Exception as e:
    print(f" MongoDB connection failed — {e}")
    db         = None
    collection = None
    fs         = None

ranker = None

try:
    if os.path.exists(PKL_LOCAL_PATH):
        print(" Loading tfidf_data.pkl from local cache...")
        with open(PKL_LOCAL_PATH, "rb") as f:
            model_data = pickle.load(f)
        print(" Loaded from local cache")
    else:
        if fs is None:
            raise Exception("MongoDB not connected — cannot download pkl")
        print(" Downloading tfidf_data.pkl from GridFS (first time only)...")
        pkl_file = fs.find_one({"filename": "tfidf_data.pkl"})
        if pkl_file is None:
            raise Exception("tfidf_data.pkl not found in GridFS")
        pkl_bytes = pkl_file.read()
        with open(PKL_LOCAL_PATH, "wb") as f:
            f.write(pkl_bytes)
        print(" Saved to local cache")
        model_data = pickle.loads(pkl_bytes)

    ranker = InfluencerRanker()
    ranker.vectorizer    = model_data["vectorizer"]
    ranker.tfidf_matrix  = model_data["tfidf_matrix"]
    ranker.influencer_df = model_data["influencer_df"]
    print(f" TF-IDF model loaded successfully ")
    print(f"   Influencers loaded : {len(ranker.influencer_df)}")

except Exception as e:
    print(f" Error loading model — {e}")
    ranker = None

def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj


def generate_ai_summary(influencer: dict, campaign: dict) -> str:

    tier      = influencer.get("tier", "nano")
    category  = influencer.get("category", "creator")
    country   = influencer.get("country", "unknown location")
    relevance = float(influencer.get("relevance_score_norm") or 0)
    engagement= float(influencer.get("engagement_score_norm") or 0)
    sentiment = float(influencer.get("sentiment_score") or 0.5)
    camp_name = campaign.get("campaign_name", "this campaign")

    if relevance > 0.7:
        rel_str = "highly relevant content"
    elif relevance > 0.4:
        rel_str = "relevant content"
    else:
        rel_str = "niche content"

    # engagement label
    if engagement > 0.7:
        eng_str = "exceptional engagement"
    elif engagement > 0.4:
        eng_str = "strong engagement"
    else:
        eng_str = "steady engagement"

    # sentiment label
    if sentiment > 0.65:
        sent_str = "a highly positive audience"
    elif sentiment > 0.45:
        sent_str = "a neutral audience"
    else:
        sent_str = "a mixed audience"

    return (
        f"A {tier}-tier {category} creator from {country} with "
        f"{rel_str}, {eng_str}, and {sent_str} — "
        f"well-suited for {camp_name}."
    )

def make_cache_key(data: dict) -> str:
    """Creates a unique key for identical campaign inputs."""
    import hashlib, json
    key = json.dumps({
        "campaign_name": data.get("campaign_name"),
        "category"     : data.get("category"),
        "location"     : data.get("location"),
        "age_min"      : data.get("age_min"),
        "age_max"      : data.get("age_max"),
        "budget"       : data.get("budget"),
        "top_n"        : data.get("top_n"),
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()

def get_campaigns_for_username(username: str):
    """Finds all campaigns an influencer appeared in + their latest data."""
    campaigns       = []
    influencer_data = None
    for doc in collection.find({"results.username": username}):
        for inf in doc.get("results", []):
            if inf.get("username") == username:
                influencer_data = inf
                break
        campaigns.append({
            "brand_name": doc.get("campaign_name", "Unknown"),
            "category"  : doc.get("category", ""),
            "location"  : doc.get("location", ""),
            "created_at": str(doc.get("created_at", ""))
        })
    return influencer_data, campaigns

def validate_inputs(data):
    errors = []

    brand = data.get("brand_name", "").strip()
    if len(brand) < 2:
        errors.append("brand_name must be at least 2 characters.")

    name = data.get("campaign_name", "").strip()
    if len(name) < 3:
        errors.append("campaign_name must be at least 3 characters.")
    if not re.match(r"^[a-zA-Z0-9\s\-&'.]+$", name):
        errors.append("campaign_name contains invalid characters.")
    if not any(c.isalpha() for c in name):
        errors.append("campaign_name must contain at least one letter.")

    desc  = data.get("description", "").strip()
    words = [w for w in desc.split() if w.isalpha()]
    if len(desc) < 20:
        errors.append("description must be at least 20 characters.")
    if len(words) < 4:
        errors.append("description must contain at least 4 real words.")
    if len(set(desc.lower().split())) < 4:
        errors.append("description must contain at least 4 unique words.")
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    if avg_word_len > 12:
        errors.append("description does not look valid.")
    for word in words:
        if len(set(word)) < len(word) * 0.4:
            errors.append("description does not look valid.")
            break

    cat = data.get("category", "").strip().lower()
    if cat not in VALID_CATEGORIES:
        errors.append(f"category must be one of: {', '.join(VALID_CATEGORIES)}")

    loc = data.get("location", "").strip()
    if not loc:
        errors.append("location is required.")
    elif not re.match(r"^[a-zA-Z\s\-]+$", loc):
        errors.append("location can only contain letters.")
    elif len(loc) < 2:
        errors.append("location must be at least 2 characters.")

    budget = data.get("budget", None)
    if budget is not None and budget != "":
        try:
            budget = float(budget)
            if budget < 100:
                errors.append("budget must be at least $100.")
        except Exception:
            errors.append("budget must be a number.")

    try:
        age_min = int(data.get("age_min", 0))
        age_max = int(data.get("age_max", 0))
        if age_min < 13 or age_min > 80:
            errors.append("age_min must be between 13 and 80.")
        if age_max < 13 or age_max > 80:
            errors.append("age_max must be between 13 and 80.")
        if age_min >= age_max:
            errors.append("age_min must be less than age_max.")
    except Exception:
        errors.append("age_min and age_max must be whole numbers.")

    try:
        top_n = int(data.get("top_n", 10))
        if top_n < 1 or top_n > 100:
            errors.append("top_n must be between 1 and 100.")
    except Exception:
        errors.append("top_n must be a whole number.")

    return errors



@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status"       : "running",
        "model_loaded" : ranker is not None,
        "db_connected" : collection is not None,
        "message"      : "Influencer Marketing Platform API"
    }), 200


@app.route("/rank", methods=["POST"])
def rank():
    try:
        if ranker is None:
            return jsonify({"error": "Model not loaded. Run precompute.py first."}), 500
        if collection is None:
            return jsonify({"error": "Database not connected."}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON body."}), 400

        errors = validate_inputs(data)
        if errors:
            return jsonify({"errors": errors}), 400

    
        cache_key = make_cache_key(data)
        if cache_key in _rank_cache:
            print(" Returning cached result")
            return jsonify(_rank_cache[cache_key]), 200

        brand_name    = data.get("brand_name", "").strip()
        campaign_name = data.get("campaign_name").strip()
        description   = data.get("description").strip()
        category      = data.get("category").strip().lower()
        location      = data.get("location").strip().lower()
        age_min       = int(data.get("age_min"))
        age_max       = int(data.get("age_max"))
        top_n         = int(data.get("top_n", 10))
        budget        = data.get("budget", None)

        if budget is not None and budget != "":
            budget = float(budget)
        else:
            budget = None

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

        if results.empty:
            return jsonify({"message": "No influencers found matching filters."}), 200

        results_list = convert_types(results.to_dict(orient="records"))

       
        campaign_context = {
            "campaign_name": campaign_name,
            "description"  : description,
            "category"     : category,
            "location"     : location,
        }
        for influencer in results_list:
            influencer["ai_summary"] = generate_ai_summary(influencer, campaign_context)

        document = {
            "brand_name"    : brand_name,
            "campaign_name" : campaign_name,
            "description"   : description,
            "category"      : category,
            "location"      : location,
            "budget"        : budget,
            "age_min"       : age_min,
            "age_max"       : age_max,
            "top_n"         : top_n,
            "created_at"    : datetime.utcnow(),
            "results"       : results_list,
        }

        inserted = collection.insert_one(document)

        response = {
            "message"        : "Ranking complete",
            "result_id"      : str(inserted.inserted_id),
            "campaign_name"  : campaign_name,
            "total_returned" : len(results_list),
            "results"        : results_list,
        }
        _rank_cache[cache_key] = response

        return jsonify(response), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results", methods=["GET"])
def get_results():
    try:
        if collection is None:
            return jsonify({"error": "Database not connected."}), 500

        results = []
        for doc in collection.find().sort("created_at", -1):
            doc["_id"] = str(doc["_id"])
            results.append(doc)

        return jsonify({"total": len(results), "results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summary", methods=["GET"])
def get_summary():
    """Lightweight endpoint — campaign metadata only, no full results payload."""
    try:
        if collection is None:
            return jsonify({"error": "Database not connected."}), 500

        results = []
        for doc in collection.find(
            {},
            {"campaign_name": 1, "brand_name": 1, "category": 1,
             "location": 1, "created_at": 1, "results": {"$slice": 1}}
        ).sort("created_at", -1):
            doc["_id"]        = str(doc["_id"])
            doc["created_at"] = str(doc.get("created_at", ""))
            results.append(doc)

        return jsonify({"total": len(results), "results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/influencer/<identifier>", methods=["GET"])
def get_influencer(identifier):

    try:
        if collection is None:
            return jsonify({"error": "Database not connected"}), 500

        influencer_data = None
        campaigns       = []

        try:
            object_id = ObjectId(identifier)
            doc = collection.find_one({"_id": object_id})

            if doc:
                index   = request.args.get("index", 0, type=int)
                results = doc.get("results", [])

                if index < 0 or index >= len(results):
                    return jsonify({"error": "Index out of range"}), 404

                username = results[index].get("username")
                influencer_data, campaigns = get_campaigns_for_username(username)

        except Exception:
            influencer_data, campaigns = get_campaigns_for_username(identifier)

        if not influencer_data:
            return jsonify({"error": "Influencer not found"}), 404

        return jsonify(convert_types({
            "influencer": influencer_data,
            "campaigns" : campaigns,
            "total"     : len(campaigns)
        })), 200

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)