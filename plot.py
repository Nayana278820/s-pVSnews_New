import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the merged dataset
merged_df = pd.read_csv('sp_500_sentiment.csv')

# Drop rows with missing values
merged_df = merged_df.dropna(subset=['Pct_Change', 'Sentiment'])

# Swapped: X = Sentiment, Y = Pct_Change
x = merged_df['Sentiment']
y = merged_df['Pct_Change']

# Linear regression
m, b = np.polyfit(x, y, 1)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5, label='Data points')
plt.plot(x, m * x + b, color='red', label=f'Regression line (y = {m:.2f}x + {b:.2f})')

# Labels and title
plt.title('News Sentiment vs S&P500 Percentage Change', fontsize=14)
plt.xlabel('News Sentiment', fontsize=12)
plt.ylabel('S&P500 % Change', fontsize=12)
plt.grid(True)
plt.legend()

# Show
plt.show()

# Optional: Correlation coefficient
corr = np.corrcoef(x, y)[0, 1]
print(f"Pearson correlation coefficient: {corr:.3f}")
