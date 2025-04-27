import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv('datasets/sp_500_sentiment.csv')

# Make sure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort data
df = df.sort_values('Date')

# Extract the year
df['Year'] = df['Date'].dt.year

# Get list of unique years
years = df['Year'].unique()

# Plot each year separately
for year in years:
    df_year = df[df['Year'] == year]
    
    plt.figure(figsize=(14, 6))  # one big figure for each year
    plt.plot(df_year['Date'], df_year['Pct_Change'], label='Pct_Change', linestyle='-', color='blue')
    plt.plot(df_year['Date'], df_year['Sentiment'], label='Sentiment', linestyle=':', color='black')
    
    plt.title(f'Pct_Change and Sentiment ({year})', fontsize=18, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # fits everything nicely within the figure
    plt.show()

# After all years, plot everything combined
plt.figure(figsize=(16, 8))
plt.plot(df['Date'], df['Pct_Change'], label='Pct_Change', linestyle='-', color='blue')
plt.plot(df['Date'], df['Sentiment'], label='Sentiment', linestyle=':', color='black')
plt.title('Pct_Change and Sentiment (All Years Combined)', fontsize=20, pad=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
