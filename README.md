# Stock Market Trends Based on News Headlines

This project explores the relationship between **news headline sentiment** and the **daily percentage change in the S&P 500 index** using historical data and basic machine learning techniques.

---

## Objective

To determine whether **daily news sentiment** has a predictive or explanatory impact on **stock market performance**, specifically the S&P 500.

---

## Files in This Repo

| File | Description |
|------|-------------|
| `news_date.py` | Cleans and reformats dates from the raw CNBC headlines, creating a clean `cleanedNews_data.csv` file. |
| `cleanedNews_data.csv` | Cleaned news headlines and descriptions, organized by date. |
| `snp_date.py` | Downloads and prepares S&P 500 stock data into `sp500_data.csv`. |
| `sp500_data.csv` | Raw S&P 500 historical data used before merging with sentiment. |
| `sentimentscore.py` | Analyzes news headline sentiment and outputs `news_with_sentiment.csv`. |
| `news_with_sentiment.csv` | Contains cleaned news headlines, descriptions, dates, and their corresponding VADER-calculated sentiment scores. |
| `merge_date.py` | Merges cleaned news sentiment scores with S&P 500 stock data based on the `Date` column, saving it as `sp_500_sentiment.csv`. |
| `sp_500_sentiment.csv` | Final merged dataset containing S&P 500 market data along with average daily news sentiment. |
| `plot.py` | Analyzes and visualizes the relationship between news sentiment and future S&P 500 price changes using clustering and correlation analysis. |
| `sp500PlusSentiment.py` | (Alternative shortcut) Calculates sentiment and merges it with stock data directly in one step (use instead of `sentimentscore.py` + `merge_date.py` if preferred). |



## Order to Run

### Full Step-by-Step (recommended):
```bash
python news_date.py
python snp_date.py
python sentimentscore.py
python merge_date.py
python plot.py
```

### Shortcut:
```bash
python news_date.py
python snp_date.py
python sp500plus.py
python plot.py
```
### *Running just plot.py should be enough to see the correlation*


---

##  Tools & Libraries Used

- `yfinance` for downloading stock data
- `pandas` for data cleaning and manipulation
- `matplotlib` for plotting
- `scikit-learn` for linear regression modeling
-  `nltk (VADER) for sentiment analysis

---


## Analysis Steps

1. **Download S&P 500 data** from 2018–2023 using `yfinance`
2. **Calculate daily % change** in closing prices
3. **Load news headlines** and compute daily average sentiment score
4. **Merge datasets** on the date column
5. **Visualize** the relationship using scatter plots
6. **Run linear regression** to measure how well sentiment predicts stock movement
7. **Cluster and correlate** news sentiment with future S&P 500 movements
8. **Evaluate** 
first graphs: using regression coefficients and R² score
second graphs: using Pearson correlation and KMeans cluster groupings


---

## Results

- A linear regression model was fitted on sentiment vs. market movement.
- The **regression line** was almost flat, and the **R² score was close to 0**, indicating little to no linear relationship.
-  **KMeans** clustering showed some grouping patterns between sentiment and future stock movement.

- **Pearson correlation** between sentiment and next week's market change was relatively low.
- **Conclusion:** Headline sentiment **alone** is not a strong predictor of S&P 500 daily returns.

---

## Future Improvements

- Include **lagged sentiment effects** (e.g., yesterday’s news on today’s market)
- Add more **features** such as headline category, source, or volume of news
- Try **non-linear models** or classification approaches
- Expand to individual stocks or sectors instead of just the index

---

## How to Run

1. Clone this repo
2. Install requirements: `pip install -r requirements.txt`
3. Run the script or Jupyter notebook

---

## Author

**Pooja Masani**  
Computer Science @ UTK  


**Nayana Patil**  
Computer Science @ UTK  

