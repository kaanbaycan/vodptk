import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# 1. Load and Clean Data
df = pd.read_csv('flight_cleaned.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df['Airline'] = df['Airline'].replace({'Indigo': 'IndiGo', 'AirAsia': 'Air Asia'})
df['From'] = df['From'].str.title()
df['To'] = df['To'].str.title()
df['Route'] = df['From'] + " -> " + df['To']

def parse_stops(val):
    s = str(val).lower()
    if 'non-stop' in s: return 0
    elif '1' in s: return 1
    elif '2' in s: return 2
    return 0

df['Stops_Num'] = df['Stops'].apply(parse_stops)

df['Booking_date_dt'] = pd.to_datetime(df['Booking_date'], format='%d-%m-%Y')
df['Journey_date_dt'] = pd.to_datetime(df['Journey_date'], format='%d-%m-%Y')

df['Journey_Day'] = df['Journey_date_dt'].dt.day
df['Journey_Month'] = df['Journey_date_dt'].dt.month
df['Journey_DayOfWeek'] = df['Journey_date_dt'].dt.dayofweek
df['Journey_IsWeekend'] = df['Journey_DayOfWeek'].isin([5, 6]).astype(int)

df['Dep_Hour'] = pd.to_datetime(df['Departure'], format='%H:%M').dt.hour
df['Dep_Minute'] = pd.to_datetime(df['Departure'], format='%H:%M').dt.minute

def parse_arrival(arr_str):
    arr_str = str(arr_str).replace('\r', '').replace('\n', ' ')
    is_next_day = 1 if '+1D' in arr_str or '+2D' in arr_str else 0
    time_part = arr_str.split()[0]
    try:
        dt = pd.to_datetime(time_part, format='%H:%M')
        return dt.hour, dt.minute, is_next_day
    except:
        return 0, 0, is_next_day

arr_parsed = df['Arrival'].apply(parse_arrival)
df['Arr_Hour'] = [x[0] for x in arr_parsed]
df['Arr_Minute'] = [x[1] for x in arr_parsed]
df['Is_Next_Day'] = [x[2] for x in arr_parsed]

def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Night'

df['Dep_TimeOfDay'] = df['Dep_Hour'].apply(get_time_of_day)
df['Arr_TimeOfDay'] = df['Arr_Hour'].apply(get_time_of_day)
df['Is_Premium_Airline'] = df['Airline'].isin(['Vistara', 'Air India']).astype(int)

cat_cols = ['Airline', 'From', 'To', 'Route', 'Booking_Slot', 'Dep_TimeOfDay', 'Arr_TimeOfDay']
num_cols = [
    'Stops_Num', 'Booking_Hour', 'Lead_Time_Days', 'Duration_mins',
    'Journey_Day', 'Journey_DayOfWeek', 'Journey_IsWeekend',
    'Dep_Hour', 'Dep_Minute', 'Arr_Hour', 'Arr_Minute',
    'Is_Next_Day', 'Is_Premium_Airline'
]

X = df[cat_cols + num_cols]
y = df['Price']

# 2. Build Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

best_lgb = lgb.LGBMRegressor(
    n_estimators=300,
    num_leaves=50,
    max_depth=12,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

full_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', best_lgb)
])

# 3. Train on Full Dataset
full_pipeline.fit(X, y)

# 4. Save Model & Metadata
model_filename = 'flight_model.joblib'
joblib.dump(full_pipeline, model_filename)

metadata = {
    'airlines': sorted(df['Airline'].unique().tolist()),
    'cities': sorted(list(set(df['From'].unique().tolist() + df['To'].unique().tolist()))),
    'routes': sorted(df['Route'].unique().tolist()),
    'booking_slots': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'cat_cols': cat_cols,
    'num_cols': num_cols
}

with open('flight_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"[SUCCESS] Trained and saved full model pipeline to '{model_filename}'.")
print(f"[SUCCESS] Saved metadata to 'flight_metadata.json'.")
