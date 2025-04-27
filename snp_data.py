import yfinance as yf
import pandas as pd

# Download historical S&P 500 data (^GSPC) using yfinance
sp500_df = yf.download('^GSPC', start='2017-12-23', end='2020-07-18')

# Reset index to get 'Date' as a column
sp500_df.reset_index(inplace=True)

# Rename columns to match your desired output
sp500_df.columns = ['Date', 'Open_^GSPC', 'High_^GSPC', 'Low_^GSPC', 'Close_^GSPC', 'Volume_^GSPC']

# Calculate percentage change
sp500_df['Pct_Change'] = sp500_df['Close_^GSPC'].pct_change() * 100

# Save to CSV
sp500_df.to_csv('sp500_data.csv', index=False)

print("Data saved successfully!")