import pandas as pd

# Load both datasets
sp500_df = pd.read_csv('sp500_data.csv')
news_df = pd.read_csv('news_with_sentiment.csv')

# Convert 'Date' to datetime format in both datasets
sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.date
news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date

# Group news data by date and calculate the average sentiment
news_grouped = news_df.groupby('Date').agg({
    'Sentiment': 'mean'  # Calculate the mean sentiment per date
}).reset_index()

# Merge the sentiment data into the sp500 dataset based on 'Date'
# This ensures the sentiment is added in a separate column
sp500_df = pd.merge(sp500_df, news_grouped[['Date', 'Sentiment']], on='Date', how='left')

# Save the updated sp500 dataset with a separate 'Sentiment' column to a new CSV
sp500_df.to_csv('sp_500_sentiment.csv', index=False)

print("✅ Sentiment added to the sp500 data. File saved as sp_500_sentiment.csv")
