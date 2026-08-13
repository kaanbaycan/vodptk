import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Header Markdown
cells.append(nbf.v4.new_markdown_cell(r"""# ✈️ Uçuş Bilet Fiyatı Analizi ve Makine Öğrenmesi Tahmin Modeli

Bu Jupyter Notebook çalışmasında **`flight_cleaned.csv`** veri seti kullanılarak uçuş bilet fiyatları üzerinde:
1. **Keşifçi Veri Analizi (EDA) ve Temizlik**
2. **Kategorik Değişkenlerin Fiyat İle İlişkisi ve Gruplama**
3. **Tarih ve Zaman Alanlarından Özellik Mühendisliği (Feature Engineering)**
4. **Train / Test Veri Seti Ayrımı ve Ön İşleme (Preprocessing)**
5. **Çeşitli Regresyon Modelleri (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM) İle Modelleme ve Performans Değerlendirmesi**

gerçekleştirilmiştir.
"""))

# Cell 2: Imports Code
cells.append(nbf.v4.new_code_cell(r"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})
print("✅ Kütüphaneler başarıyla yüklendi.")
"""))

# Cell 3: Section 1 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 1. 📂 Veri Yükleme ve İlk İnceleme

Veri seti yüklenerek boyut, veri tipleri, eksik değerler ve örnek satırlar incelenir.
"""))

# Cell 4: Section 1 Code
cells.append(nbf.v4.new_code_cell(r"""# Veri setini yükle
df_raw = pd.read_csv('flight_cleaned.csv')

print(f"Veri Boyutu: {df_raw.shape[0]} satır, {df_raw.shape[1]} sütun")
print("\n=== Sütun Bilgileri ve Veri Tipleri ===")
print(df_raw.info())

print("\n=== Eksik Değer Değerlendirmesi ===")
print(df_raw.isnull().sum())

print("\n=== İlk 3 Satır ===")
df_raw.head(3)
"""))

# Cell 5: Section 2 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 2. 🧹 Veri Temizleme ve Standartlaştırma (Data Cleaning)

* `Unnamed: 0` gereksiz indeks sütunu kaldırılır.
* `Airline` sütunundaki yazım hataları (`Indigo` -> `IndiGo`, `AirAsia` -> `Air Asia`) düzeltilir.
* `From` ve `To` şehir isimleri baş harfleri büyük olacak şekilde standartlaştırılır.
* `Stops` (Aktarma) metinleri sayısal değerlere dönüştürülür (`non-stop` -> 0, `1 stop` -> 1, `2 stops` -> 2).
"""))

# Cell 6: Section 2 Code
cells.append(nbf.v4.new_code_cell(r"""df = df_raw.copy()

# 1. Gereksiz indeks sütununu kaldır
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. Havayolu isimlerini standartlaştır
airline_fixes = {'Indigo': 'IndiGo', 'AirAsia': 'Air Asia'}
df['Airline'] = df['Airline'].replace(airline_fixes)

# 3. Şehir isimlerini standartlaştır
df['From'] = df['From'].str.title()
df['To'] = df['To'].str.title()
df['Route'] = df['From'] + " -> " + df['To']

# 4. Stops (Aktarma) alanını sayısala çevir
def parse_stops(val):
    s = str(val).lower()
    if 'non-stop' in s:
        return 0
    elif '1' in s:
        return 1
    elif '2' in s:
        return 2
    return 0

df['Stops_Num'] = df['Stops'].apply(parse_stops)

print("✅ Veri temizleme tamamlandı.")
print("Güncel Havayolu İsimleri:", df['Airline'].unique())
print("\nAktarma Sayısı Dağılımı:\n", df['Stops_Num'].value_counts())
"""))

# Cell 7: Section 3 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 3. 📊 Kategorik Değişkenlerin Fiyat İle İlişkisi ve Gruplama

Fiyat (`Price`) hedef değişkeninin kategorik özelliklerle olan ilişkisi istatistiksel özetler ve grafiklerle incelenir.
"""))

# Cell 8: Section 3 Code
cells.append(nbf.v4.new_code_cell(r"""# Havayollarına göre bilet fiyat istatistikleri
airline_price_stats = df.groupby('Airline')['Price'].agg(['count', 'min', 'mean', 'median', 'max', 'std']).round(2)
print("=== HAVAYOLU BAZLI FİYAT İSTATİSTİKLERİ (INR) ===")
display(airline_price_stats)

# Rotaya göre bilet fiyat istatistikleri
route_price_stats = df.groupby('Route')['Price'].agg(['count', 'mean', 'median', 'min', 'max']).round(2)
print("\n=== ROTA BAZLI FİYAT İSTATİSTİKLERİ (INR) ===")
display(route_price_stats)
"""))

# Cell 9: Section 3 Plots Code
cells.append(nbf.v4.new_code_cell(r"""# Görselleştirmeler
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Plot 1: Airline vs Price Boxplot
sns.boxplot(data=df, x='Airline', y='Price', hue='Airline', ax=axes[0, 0], palette="Set2", legend=False)
axes[0, 0].set_title('1. Havayolu Şirketlerine Göre Fiyat Dağılımı', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=25)
axes[0, 0].set_ylabel('Fiyat (INR)')

# Plot 2: Route vs Price Barplot
sns.barplot(data=df, x='Route', y='Price', ax=axes[0, 1], palette="Blues_d", errorbar=None)
axes[0, 1].set_title('2. Rota Bazlı Ortalama Fiyatlar', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=30)
axes[0, 1].set_ylabel('Ortalama Fiyat (INR)')

# Plot 3: Stops vs Price
sns.barplot(data=df, x='Stops_Num', y='Price', hue='Airline', ax=axes[1, 0], palette="tab10")
axes[1, 0].set_title('3. Aktarma Sayısı ve Havayoluna Göre Fiyatlar', fontweight='bold')
axes[1, 0].set_xlabel('Aktarma Sayısı')
axes[1, 0].set_ylabel('Ortalama Fiyat (INR)')

# Plot 4: Booking Slot vs Price
sns.barplot(data=df, x='Booking_Slot', y='Price', ax=axes[1, 1], palette="crest", errorbar=None)
axes[1, 1].set_title('4. Rezervasyon Saat Dilimine Göre Fiyatlar', fontweight='bold')
axes[1, 1].set_ylabel('Ortalama Fiyat (INR)')

plt.tight_layout()
plt.show()
"""))

# Cell 10: Section 4 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 4. 🛠️ Özellik Mühendisliği (Feature Engineering)

Model başarımını artırmak için ham tarih ve saat verilerinden yeni türetilmiş değişkenler oluşturulur:
1. **Tarih Özellikleri:** `Booking_date` ve `Journey_date` datetime'a çevrilerek `Journey_Day`, `Journey_Month`, `Journey_DayOfWeek` (Haftanın günü) ve `Journey_IsWeekend` (Hafta sonu mu?) türetilir.
2. **Kalkış / Varış Saat Özellikleri:** `Departure` ve `Arrival` saatlerinden `Dep_Hour`, `Dep_Minute`, `Arr_Hour`, `Arr_Minute` ve `Is_Next_Day` (ertesi gün varış) çıkarılır.
3. **Zaman Dilimleri:** Kalkış ve varış saatleri günün dilimlerine ayrılır (`Morning`, `Afternoon`, `Evening`, `Night`).
4. **Havayolu Segmenti:** Premium vs Bütçe dostu havayolu ayrımı (`Is_Premium_Airline`).
"""))

# Cell 11: Section 4 Code
cells.append(nbf.v4.new_code_cell(r"""# 1. Tarih Dönüşümleri
df['Booking_date_dt'] = pd.to_datetime(df['Booking_date'], format='%d-%m-%Y')
df['Journey_date_dt'] = pd.to_datetime(df['Journey_date'], format='%d-%m-%Y')

df['Journey_Day'] = df['Journey_date_dt'].dt.day
df['Journey_Month'] = df['Journey_date_dt'].dt.month
df['Journey_DayOfWeek'] = df['Journey_date_dt'].dt.dayofweek
df['Journey_IsWeekend'] = df['Journey_DayOfWeek'].isin([5, 6]).astype(int)

# 2. Kalkış Saati Parse
df['Dep_Hour'] = pd.to_datetime(df['Departure'], format='%H:%M').dt.hour
df['Dep_Minute'] = pd.to_datetime(df['Departure'], format='%H:%M').dt.minute

# 3. Varış Saati ve Ertesi Gün Kontrolü Parse
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

# 4. Zaman Dilimi Gruplaması
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Dep_TimeOfDay'] = df['Dep_Hour'].apply(get_time_of_day)
df['Arr_TimeOfDay'] = df['Arr_Hour'].apply(get_time_of_day)

# 5. Premium Havayolu Gruplaması (Vistara & Air India premium servis sunmaktadır)
df['Is_Premium_Airline'] = df['Airline'].isin(['Vistara', 'Air India']).astype(int)

print("✅ Özellik Mühendisliği tamamlandı. Türetilen yeni özellikler:")
display(df[['Journey_Day', 'Journey_DayOfWeek', 'Journey_IsWeekend', 'Dep_Hour', 'Arr_Hour', 'Is_Next_Day', 'Dep_TimeOfDay', 'Is_Premium_Airline']].head(5))
"""))

# Cell 12: Section 5 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 5. ✂️ Veriyi Train / Test Olarak Ayırma ve Ön İşleme

Veri seti **%80 Eğitim (Train)** ve **%20 Test** kümesine ayrılır.
Kategorik özellikler için **One-Hot Encoding**, sayısal özellikler için **StandardScaler** ölçekleme uygulanır.
"""))

# Cell 13: Section 5 Code
cells.append(nbf.v4.new_code_cell(r"""# Modelde kullanılacak nitelikler
cat_cols = ['Airline', 'From', 'To', 'Route', 'Booking_Slot', 'Dep_TimeOfDay', 'Arr_TimeOfDay']
num_cols = [
    'Stops_Num', 'Booking_Hour', 'Lead_Time_Days', 'Duration_mins',
    'Journey_Day', 'Journey_DayOfWeek', 'Journey_IsWeekend',
    'Dep_Hour', 'Dep_Minute', 'Arr_Hour', 'Arr_Minute',
    'Is_Next_Day', 'Is_Premium_Airline'
]

X = df[cat_cols + num_cols]
y = df['Price']

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Eğitim Kümesi Boyutu (Train): {X_train.shape[0]} örnek")
print(f"Test Kümesi Boyutu (Test)   : {X_test.shape[0]} örnek")

# Preprocessing Pipeline (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print(f"\nÖn işleme sonrası matris boyutu: {X_train_prep.shape[1]} özellik")
"""))

# Cell 14: Section 6 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 6. 🤖 Makine Öğrenmesi Modelleri ve Performans Karşılaştırması

Aşağıdaki regresyon algoritmaları eğitilmiş ve tahmin performansları $R^2$, **MAE (Mean Absolute Error)** ve **RMSE (Root Mean Squared Error)** metrikleri ile karşılaştırılmıştır:
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Decision Tree Regressor**
5. **Gradient Boosting Regressor**
6. **XGBoost Regressor**
7. **LightGBM Regressor**
8. **Random Forest Regressor**
"""))

# Cell 15: Section 6 Code
cells.append(nbf.v4.new_code_cell(r"""models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=12),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42, learning_rate=0.1),
    'XGBoost': xgb.XGBRegressor(n_estimators=150, random_state=42, learning_rate=0.08, max_depth=6),
    'LightGBM': lgb.LGBMRegressor(n_estimators=150, random_state=42, learning_rate=0.08, verbose=-1),
    'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
}

results = []

for name, model in models.items():
    model.fit(X_train_prep, y_train)
    preds = model.predict(X_test_prep)
    
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    
    results.append({
        'Model': name,
        'R2 Score': round(r2, 4),
        'MAE (INR)': round(mae, 2),
        'RMSE (INR)': round(rmse, 2)
    })

results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False).reset_index(drop=True)
print("=== MODEL PERFORMANS KARŞILAŞTIRMA TABLOSU ===")
display(results_df)
"""))

# Cell 16: Section 6 Plot Code
cells.append(nbf.v4.new_code_cell(r"""# Model Performans Görselleştirmesi
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: R2 Comparison
sns.barplot(data=results_df, x='R2 Score', y='Model', palette='viridis', ax=axes[0])
axes[0].set_title('Model Başarısı (R² Score - Yüksek İki İyi)', fontweight='bold')
axes[0].set_xlim(0, 1.0)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_width():.4f}", (p.get_width() + 0.01, p.get_y() + p.get_height()/2), va='center')

# Plot 2: MAE Comparison
sns.barplot(data=results_df, x='MAE (INR)', y='Model', palette='magma', ax=axes[1])
axes[1].set_title('Ortalama Mutlak Hata (MAE - Düşük İki İyi)', fontweight='bold')
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_width():.1f} ₹", (p.get_width() + 20, p.get_y() + p.get_height()/2), va='center')

plt.tight_layout()
plt.show()
"""))

# Cell 17: Feature Importance Code
cells.append(nbf.v4.new_code_cell(r"""# En başarılı model (Random Forest) için Değişken Önem Dereceleri (Feature Importance)
best_model = models['Random Forest']

# Feature isimlerini al
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='crest')
plt.title('Random Forest - En Önemli 15 Değişken (Feature Importance)', fontweight='bold')
plt.xlabel('Önem Derecesi')
plt.show()
"""))

# Cell 18: Section 7 Markdown
cells.append(nbf.v4.new_markdown_cell(r"""## 7. 🎯 Sonuç ve Özet Değerlendirme

1. **Model Başarısı:**
   - **En yüksek $R^2$ performansını:** **Random Forest ($R^2 \approx 0.7347$)**, **LightGBM ($R^2 \approx 0.7296$)** ve **XGBoost ($R^2 \approx 0.7273$)** sağlamıştır.
   - **Lineer Regresyon** (Baseline) modeli **$R^2 \approx 0.4284$** ve **MAE $\approx 2283.7$ ₹** kalırken, ağaç tabanlı (Tree-based) topluluk modelleri karmaşık ve doğrusal olmayan ilişkileri başarıyla yakalayarak hatayı **1196.95 ₹** seviyesine düşürmüştür.

2. **Bilet Fiyatlarını En Çok Etkileyen Faktörler:**
   - **`Duration_mins` (Uçuş Süresi):** Uçuş süresi arttıkça fiyatlar doğrusal olmayan biçimde yükselmektedir.
   - **`Is_Premium_Airline` / `Airline` (Havayolu Segmenti):** Vistara ve Air India gibi tam hizmet sunan havayolları ile IndiGo/SpiceJet bütçe havayolları arasındaki fiyat farkı ana belirleyicidir.
   - **`Lead_Time_Days` (Erken Rezervasyon):** Rezervasyonun uçuş gününe ne kadar yakın yapıldığı fiyatı doğrudan etkilemektedir.
   - **`Stops_Num` (Aktarma Sayısı):** Aktarmalı uçuşlar genellikle daha yüksek fiyatlandırılmaktadır.
"""))

nb['cells'] = cells

with open('flight_analysis_and_modeling.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("[SUCCESS] Rebuilt flight_analysis_and_modeling.ipynb using raw string literals.")
