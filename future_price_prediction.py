import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your merged dataset
df = pd.read_csv('datasets/sp_500_sentiment.csv')

# Quick data peek
print(df.head())

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])

# Create "target" column: 1 if next day's Pct_Change is positive, else 0
df['Next_Day_Pct_Change'] = df['Pct_Change'].shift(-1)  # shift to predict next day
df['UpDown'] = (df['Next_Day_Pct_Change'] > 0).astype(int)

# Create 5-day moving average of Sentiment
df['Sentiment_MA5'] = df['Sentiment'].rolling(window=5).mean()

# Drop rows with NaN (because moving average needs 5 data points)
df = df.dropna()

# Features: 5-day moving average of Sentiment
X = df[['Sentiment_MA5']]
y = df['UpDown']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)  # keep time order

# Build and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy with Sentiment MA5: {acc:.4f}")