import pandas as pd

# Load the datasets
sp500_df = pd.read_csv('sp500_data.csv')
news_df = pd.read_csv('cleanedNews_data.csv')

# Convert 'Date' column to datetime format in both datasets to ensure proper alignment
sp500_df['Date'] = pd.to_datetime(sp500_df['Date']).dt.date
news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date

# Group news data by date and calculate the average sentiment
news_grouped = news_df.groupby('Date').agg({
    'Sentiment': 'mean'  # Calculate the mean sentiment per date
}).reset_index()

# Merge sentiment data into sp500 data, using the 'Date' column
sp500_df = pd.merge(sp500_df, news_grouped[['Date', 'Sentiment']], on='Date', how='left')

# Save the merged data to a new CSV with sentiment
sp500_df.to_csv('sp_500_sentiment.csv', index=False)

print("✅ Sentiment successfully aligned and saved as sp_500_sentiment.csv")
