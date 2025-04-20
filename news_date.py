import pandas as pd
import re
from datetime import datetime

# Load the original CSV
df = pd.read_csv('cnbc_headlines.csv')

# Month abbreviation to full month mapping (including "Sept")
month_mapping = {
    'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
    'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
    'Sep': 'September', 'Sept': 'September', 'Oct': 'October',
    'Nov': 'November', 'Dec': 'December'
}

# Function to extract and format date
def extract_date(time_str):
    try:
        match = re.search(r'(\d{1,2}:\d{2})\s+[AP]M\s+ET\s+\w+,\s+(\d{1,2})\s+(\w+)\s+(\d{4})', time_str.strip())
        if match:
            day = int(match.group(2))
            month_str = match.group(3)
            year = int(match.group(4))

            # Convert abbreviated or malformed months to full month names
            month_str_cap = month_str.capitalize()
            month_str = month_mapping.get(month_str_cap, month_str_cap)

            date_obj = datetime.strptime(f"{day} {month_str} {year}", "%d %B %Y")
            return date_obj.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error processing time string '{time_str.strip()}': {e}")
        return None

# Apply function to 'Time' column
df['Date'] = df['Time'].apply(lambda x: extract_date(x) if pd.notnull(x) else None)

# Save cleaned data
df.to_csv('cleanedNews_data.csv', index=False)

print("✅ New CSV file 'cleanedNews_data.csv' created with reformatted dates.")
