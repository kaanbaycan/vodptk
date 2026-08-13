import pandas as pd
import numpy as np

# Load original airports dataset
df = pd.read_csv('airports.csv')

# Filter for India (iso_country == 'IN')
df_in = df[df['iso_country'] == 'IN'].copy()

# Sort order for airport types
type_priority = {
    'large_airport': 1,
    'medium_airport': 2,
    'small_airport': 3,
    'seaplane_base': 4,
    'heliport': 5,
    'closed': 6
}
df_in['type_order'] = df_in['type'].map(type_priority).fillna(99)

# Clean string fields
df_in['name'] = df_in['name'].str.strip()
df_in['municipality'] = df_in['municipality'].str.strip().str.title()

# State Mapping from iso_region (IN-XX)
state_map = {
    'IN-AN': 'Andaman and Nicobar Islands',
    'IN-AP': 'Andhra Pradesh',
    'IN-AR': 'Arunachal Pradesh',
    'IN-AS': 'Assam',
    'IN-BR': 'Bihar',
    'IN-CH': 'Chandigarh',
    'IN-CT': 'Chhattisgarh',
    'IN-DH': 'Dadra and Nagar Haveli and Daman and Diu',
    'IN-DL': 'Delhi',
    'IN-GA': 'Goa',
    'IN-GJ': 'Gujarat',
    'IN-HP': 'Himachal Pradesh',
    'IN-HR': 'Haryana',
    'IN-JH': 'Jharkhand',
    'IN-JK': 'Jammu and Kashmir',
    'IN-KA': 'Karnataka',
    'IN-KL': 'Kerala',
    'IN-LA': 'Ladakh',
    'IN-LD': 'Lakshadweep',
    'IN-MH': 'Maharashtra',
    'IN-ML': 'Meghalaya',
    'IN-MN': 'Manipur',
    'IN-MP': 'Madhya Pradesh',
    'IN-MZ': 'Mizoram',
    'IN-NL': 'Nagaland',
    'IN-OR': 'Odisha',
    'IN-PB': 'Punjab',
    'IN-PY': 'Puducherry',
    'IN-RJ': 'Rajasthan',
    'IN-SK': 'Sikkim',
    'IN-TN': 'Tamil Nadu',
    'IN-TR': 'Tripura',
    'IN-TS': 'Telangana',
    'IN-UK': 'Uttarakhand',
    'IN-UP': 'Uttar Pradesh',
    'IN-WB': 'West Bengal'
}

df_in['state_name'] = df_in['iso_region'].map(state_map).fillna(df_in['iso_region'])

# Sort dataframe
df_in = df_in.sort_values(by=['type_order', 'state_name', 'name']).reset_index(drop=True)

# Select and rename columns for clarity
cols_export = [
    'id', 'ident', 'name', 'type', 'scheduled_service',
    'municipality', 'state_name', 'iso_region', 'latitude_deg', 'longitude_deg',
    'elevation_ft', 'gps_code', 'local_code'
]

df_export = df_in[cols_export]

# Export to CSV
output_path = 'airports_india.csv'
df_export.to_csv(output_path, index=False)

print(f"[SUCCESS] Processed {len(df_export)} India airports and saved to '{output_path}'.")
print("\n--- Summary by Airport Type ---")
print(df_export['type'].value_counts())

print("\n--- Top Scheduled Service Large/Medium Airports ---")
commercial = df_export[df_export['scheduled_service'] == 'yes']
print(commercial[['ident', 'name', 'municipality', 'state_name', 'type']].head(15))
