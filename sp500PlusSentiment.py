import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Make sure VADER is downloaded
# nltk.download('vader_lexicon')  # Uncomment this line only once

# Load the datasets
sp500_df = pd.read_csv('datasets/sp500_data.csv')
news_df = pd.read_csv('datasets/cleanedNews_data.csv')

# Convert 'Date' column to datetime format
sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.date
news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Create a new Sentiment column based on Headlines
news_df['Sentiment'] = news_df['Headlines'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Check whats there now
print(news_df[['Headlines', 'Sentiment']].head())

# Group news data by date and calculate the average sentiment
news_grouped = news_df.groupby('Date').agg({
    'Sentiment': 'mean' 
}).reset_index()

# Merge sentiment data into sp500 data
sp500_df = pd.merge(sp500_df, news_grouped[['Date', 'Sentiment']], on='Date', how='left')

# Save the merged data
sp500_df.to_csv('datasets/sp_500_sentiment.csv', index=False)

print("Sentiment successfully calculated, aligned, and saved as sp_500_sentiment.csv")
