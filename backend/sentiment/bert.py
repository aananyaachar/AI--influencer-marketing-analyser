import pandas as pd
from transformers import pipeline
import torch

df = pd.read_csv("translated_instagram_comments.csv")

print("Dataset Loaded")
print("Shape:", df.shape)

print("Loading BERT model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

print("Model Loaded")

def convert_label(label):
    stars = int(label[0])  # "4 stars" → 4
    
    if stars >= 4:
        return "Positive"
    elif stars == 3:
        return "Neutral"
    else:
        return "Negative"

print("Running BERT sentiment...")

bert_results = []

for i, text in enumerate(df["translated_comment"]):
    try:
        result = sentiment_pipeline(str(text))[0]
        sentiment = convert_label(result['label'])
        bert_results.append(sentiment)
        
        if i % 500 == 0:
            print(f"Processed {i} comments")
            
    except:
        bert_results.append("Neutral")

df["bert_sentiment"] = bert_results
print("BERT Sentiment Completed")
print(df["bert_sentiment"].value_counts())
df = df.rename(columns={"influencer_username": "username"})
df.to_csv("final_with_bert_sentiment.csv", index=False)
print("File saved as final_with_bert_sentiment.csv")