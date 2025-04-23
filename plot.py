# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Load the merged dataset
# merged_df = pd.read_csv('sp_500_sentiment.csv')

# # Drop rows with missing values
# merged_df = merged_df.dropna(subset=['Pct_Change', 'Sentiment'])

# # Prepare data for clustering
# X = merged_df[['Sentiment', 'Pct_Change']].values

# # Optional: Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Choose number of clusters (you can tweak this)
# k = 3
# kmeans = KMeans(n_clusters=k, random_state=42)
# merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

# # Plot clusters
# plt.figure(figsize=(10, 6))
# colors = ['red', 'green', 'blue', 'purple', 'orange']

# for i in range(k):
#     cluster = merged_df[merged_df['Cluster'] == i]
#     plt.scatter(cluster['Sentiment'], cluster['Pct_Change'], 
#                 label=f'Cluster {i}', alpha=0.6, color=colors[i % len(colors)])

# # Plot centroids (convert back to original scale)
# centroids = scaler.inverse_transform(kmeans.cluster_centers_)
# plt.scatter(centroids[:, 0], centroids[:, 1], 
#             s=200, c='black', marker='X', label='Centroids')

# # Labels and title
# plt.title('K-Means Clustering: News Sentiment vs S&P500 % Change', fontsize=14)
# plt.xlabel('News Sentiment', fontsize=12)
# plt.ylabel('S&P500 % Change', fontsize=12)
# plt.legend()
# plt.grid(True)

# # Show
# plt.show()

# # Pearson correlation coefficient
# corr = np.corrcoef(merged_df['Sentiment'], merged_df['Pct_Change'])[0, 1]
# print(f"Pearson correlation coefficient: {corr:.3f}")

# # NEW: Sentiment Polarity Analysis
# positive = merged_df[merged_df['Sentiment'] > 0]
# negative = merged_df[merged_df['Sentiment'] < 0]
# neutral = merged_df[merged_df['Sentiment'] == 0]

# print("\nSentiment Polarity Group Statistics:")
# print(f"Positive Sentiment Count: {len(positive)} | Avg % Change: {positive['Pct_Change'].mean():.3f}")
# print(f"Negative Sentiment Count: {len(negative)} | Avg % Change: {negative['Pct_Change'].mean():.3f}")
# print(f"Neutral Sentiment Count: {len(neutral)} | Avg % Change: {neutral['Pct_Change'].mean():.3f}")


# lagged sentiment:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and clean data
merged_df = pd.read_csv('sp_500_sentiment.csv')
merged_df = merged_df.dropna(subset=['Pct_Change', 'Sentiment'])

# Shift % Change upward to align tomorrow's change with today's sentiment
merged_df['Next_Day_Change'] = merged_df['Pct_Change'].shift(-7)
merged_df = merged_df.dropna(subset=['Next_Day_Change'])

# Clustering on Sentiment vs NEXT day's % change
X = merged_df[['Sentiment', 'Next_Day_Change']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
merged_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot lagged sentiment clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue', 'purple', 'orange']

for i in range(k):
    cluster = merged_df[merged_df['Cluster'] == i]
    plt.scatter(cluster['Sentiment'], cluster['Next_Day_Change'], 
                label=f'Cluster {i}', alpha=0.6, color=colors[i % len(colors)])

centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], 
            s=200, c='black', marker='X', label='Centroids')

plt.title('K-Means: Sentiment vs Next Day S&P500 % Change', fontsize=14)
plt.xlabel('News Sentiment (Day t)', fontsize=12)
plt.ylabel('S&P500 % Change (Day t+1)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Pearson correlation for lagged relationship
corr_lag = np.corrcoef(merged_df['Sentiment'], merged_df['Next_Day_Change'])[0, 1]
print(f"Lagged Pearson correlation (Sentiment → Next Day % Change): {corr_lag:.3f}")

# Sentiment polarity group analysis (based on lagged target)
positive = merged_df[merged_df['Sentiment'] > 0]
negative = merged_df[merged_df['Sentiment'] < 0]
neutral = merged_df[merged_df['Sentiment'] == 0]

print("\nLagged Sentiment Polarity Stats (t sentiment → t+1 price):")
print(f"Positive Sentiment Count: {len(positive)} | Avg Next Day % Change: {positive['Next_Day_Change'].mean():.3f}")
print(f"Negative Sentiment Count: {len(negative)} | Avg Next Day % Change: {negative['Next_Day_Change'].mean():.3f}")
print(f"Neutral Sentiment Count: {len(neutral)} | Avg Next Day % Change: {neutral['Next_Day_Change'].mean():.3f}")

# Plot bar chart of average % change by sentiment polarity
avg_changes = [
    positive['Pct_Change'].mean(),
    neutral['Pct_Change'].mean(),
    negative['Pct_Change'].mean()
]
labels = ['Positive', 'Neutral', 'Negative']

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, avg_changes, color=['green', 'gray', 'red'], alpha=0.7)

# Add value labels on top
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom')

plt.title('Average S&P500 % Change 7 Days After News Sentiment')
plt.ylabel('% Change')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(min(avg_changes) - 0.5, max(avg_changes) + 0.5)

plt.show()
