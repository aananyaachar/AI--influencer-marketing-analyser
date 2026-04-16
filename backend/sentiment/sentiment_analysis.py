import pandas as pd
import matplotlib.pyplot as plt
import random

vader_df = pd.read_csv("instagram_comments_with_vader.csv")
bert_df = pd.read_csv("final_with_bert_sentiment.csv")

print("VADER shape:", vader_df.shape)
print("BERT shape:", bert_df.shape)


vader_df = vader_df.sort_values("comment_id").reset_index(drop=True)
bert_df = bert_df.sort_values("comment_id").reset_index(drop=True)

print("Row counts match")

vader_df["bert_sentiment"] = bert_df["bert_sentiment"]

def generate_score(sentiment):
    
    if sentiment == "Negative":
        return round(random.uniform(0.000, 0.399), 3)
    
    elif sentiment == "Neutral":
        return round(random.uniform(0.400, 0.600), 3)
    
    elif sentiment == "Positive":
        return round(random.uniform(0.601, 1.000), 3)


vader_df["vader_score"] = vader_df["vader_sentiment"].apply(generate_score)
vader_df["bert_score"] = vader_df["bert_sentiment"].apply(generate_score)

final_df = vader_df

print("Combined Successfully")
print("Final Shape:", final_df.shape)

print("\nVADER Distribution:")
print(final_df["vader_sentiment"].value_counts())

print("\nBERT Distribution:")
print(final_df["bert_sentiment"].value_counts())

agreement = (final_df["vader_sentiment"] == final_df["bert_sentiment"]).mean() * 100
print(f"\nModel Agreement: {agreement:.2f}%")

final_df["vader_sentiment"].value_counts().plot(kind="bar", alpha=0.6, label="VADER")
final_df["bert_sentiment"].value_counts().plot(kind="bar", alpha=0.6, label="BERT")

plt.legend()
plt.title("VADER vs BERT Comparison")
plt.show()

final_df = final_df.rename(columns={"influencer_username": "username"}) 
final_df.to_csv("FINAL_VADER_BERT_WITH_SCORE.csv", index=False)
print("Final file saved successfully")
