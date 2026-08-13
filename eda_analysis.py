import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.titlesize': 14})

# 1. Load Data
flight_df = pd.read_csv('flight_cleaned.csv')
airports_df = pd.read_csv('airports.csv')

print("=== INITIAL SHAPE ===")
print(f"Flight Data: {flight_df.shape}")
print(f"Airports Data: {airports_df.shape}")

# 2. Data Cleaning
# Clean Airline names
airline_map = {
    'Indigo': 'IndiGo',
    'AirAsia': 'Air Asia'
}
flight_df['Airline'] = flight_df['Airline'].replace(airline_map)

# Clean City names (capitalize properly)
flight_df['From'] = flight_df['From'].str.title()
flight_df['To'] = flight_df['To'].str.title()
flight_df['Route'] = flight_df['From'] + " -> " + flight_df['To']

# Save cleaned dataframe version
flight_df.to_csv('flight_processed.csv', index=False)

# 3. Summary Statistics
print("\n=== AIRLINE DISTRIBUTION ===")
print(flight_df['Airline'].value_counts())

print("\n=== ROUTE DISTRIBUTION ===")
print(flight_df['Route'].value_counts())

print("\n=== PRICE METRICS BY AIRLINE ===")
price_by_airline = flight_df.groupby('Airline')['Price'].agg(['count', 'min', 'mean', 'median', 'max', 'std']).round(2)
print(price_by_airline)

print("\n=== PRICE METRICS BY ROUTE ===")
price_by_route = flight_df.groupby('Route')['Price'].agg(['count', 'mean', 'median', 'min', 'max']).round(2)
print(price_by_route)

# 4. Airport Geo Merge Exploration
indian_airports = airports_df[
    (airports_df['iso_country'] == 'IN') & 
    (airports_df['type'].isin(['large_airport', 'medium_airport']))
][['name', 'municipality', 'iata_code' if 'iata_code' in airports_df.columns else 'ident', 'latitude_deg', 'longitude_deg']]

print("\n=== INDIAN AIRPORTS SAMPLE ===")
print(indian_airports.head())

# 5. Generate Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Chart 1: Price Distribution by Airline
sns.boxplot(data=flight_df, x='Airline', y='Price', hue='Airline', ax=axes[0, 0], palette="Set2", legend=False)
axes[0, 0].set_title('Havayolu Sirketlerine Gore Bilet Fiyati Dagilimi (INR)', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=30)
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Fiyat (INR)')

# Chart 2: Lead Time vs Price
lead_price = flight_df.groupby(['Lead_Time_Days', 'Airline'])['Price'].mean().reset_index()
sns.lineplot(data=lead_price, x='Lead_Time_Days', y='Price', hue='Airline', marker='o', ax=axes[0, 1], palette="tab10")
axes[0, 1].set_title('Erken Rezervasyon Gunu (Lead Time) vs Ortalama Fiyat', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Kac Gun Once Alindi (Lead Time Days)')
axes[0, 1].set_ylabel('Ortalama Fiyat (INR)')

# Chart 3: Route vs Price by Stop Count
sns.barplot(data=flight_df, x='Route', y='Price', hue='Stops', ax=axes[1, 0], palette="viridis")
axes[1, 0].set_title('Rotaya ve Aktarma Sayisina Gore Bilet Fiyatlari', fontsize=12, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=30)
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Ortalama Fiyat (INR)')

# Chart 4: Duration vs Price Scatter
sns.scatterplot(data=flight_df, x='Duration_mins', y='Price', hue='Airline', alpha=0.7, ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title('Ucus Suresi (Dk) vs Bilet Fiyati', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Ucus Suresi (Dakika)')
axes[1, 1].set_ylabel('Fiyat (INR)')

plt.tight_layout()
plt.savefig('eda_summary_charts.png', dpi=300)
print("\n[SUCCESS] Charts saved to eda_summary_charts.png")
