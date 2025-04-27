import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load your CSV
news_df = pd.read_csv('datasets/cleanedNews_data.csv')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Compute sentiment scores using the 'Headlines' column
news_df['Sentiment'] = news_df['Headlines'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Save the result to a new CSV
news_df.to_csv('datasets/news_with_sentiment.csv', index=False)

print("Sentiment scores added! Saved as news_with_sentiment.csv")

