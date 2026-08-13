# ✈️ Uçuş Bilet Fiyatı Analizi ve Hiperparametre Optimizasyonlu Makine Öğrenmesi Tahmin Modeli (VODP)

Bu proje, Hindistan içi iç hat uçuş veri seti (`flight_cleaned.csv`) üzerinde gerçekleştirilen **Keşifçi Veri Analizi (EDA)**, **Özellik Mühendisliği (Feature Engineering)**, **Topluluk Regresyon Modelleri** ve **Hiperparametre Optimizasyonu (RandomizedSearchCV)** çalışmasını içermektedir.

---

## ⚡ Makine Öğrenmesi & Hiperparametre Optimizasyon Performansı

Tüm modeller **%80 Train / %20 Test** ayrımı ve ön işleme pipeline'ı (`StandardScaler` + `OneHotEncoder`) ile değerlendirilmiştir:

### 🏆 Modellerin Karşılaştırmalı Başarısı

| Model | Tip / Durum | $R^2$ Score (Açıklanan Varyans) | MAE (Ortalama Mutlak Hata) | RMSE (Kök Ortalama Kare Hata) |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **LightGBM** | **Optimizasyonlu (Tuned)** | **0.7566** | **1,219.10 ₹** | **2,087.65 ₹** |
| 🥈 **Random Forest** | **Optimizasyonlu (Tuned)** | **0.7370** | **1,195.77 ₹** | **2,170.08 ₹** |
| 🥉 **Random Forest** | Varsayılan (Default) | **0.7347** | **1,196.95 ₹** | **2,179.68 ₹** |
| 🔹 **LightGBM** | Varsayılan (Default) | **0.7296** | **1,311.34 ₹** | **2,200.48 ₹** |
| 🔹 **XGBoost** | Varsayılan (Default) | **0.7273** | **1,278.95 ₹** | **2,209.80 ₹** |
| 🔹 **Gradient Boosting** | Varsayılan (Default) | **0.6494** | **1,612.46 ₹** | **2,505.72 ₹** |
| 🔹 **Decision Tree** | Varsayılan (Default) | **0.4856** | **1,467.32 ₹** | **3,035.24 ₹** |
| 🔸 **Linear Regression (Baseline)** | Baseline | **0.4284** | **2,283.70 ₹** | **3,199.53 ₹** |

---

## 🛠️ Hiperparametre Optimizasyon Süreci

3-Fold Cross-Validation ve `RandomizedSearchCV` ile en iyi parametreler:
- **LightGBM (Tuned):** `n_estimators: 300`, `num_leaves: 50`, `max_depth: 12`, `learning_rate: 0.1` -> **$R^2 = 0.7566$** (Varsayılan 0.7296'dan önemli artış).
- **Random Forest (Tuned):** `n_estimators: 300`, `min_samples_split: 2`, `min_samples_leaf: 1`, `max_depth: None` -> **$R^2 = 0.7370$**.

---

## 🔑 Önemli Çıkarımlar & Bulgular

1. **Hiperparametre Tuning Başarısı:** Optimizasyon sayesinde LightGBM modeli $R^2$ skorunu **%75.7** seviyesine yükselterek en iyi genel model olmuştur.
2. **Lineer Regresyon vs Topluluk Modelleri:** Lineer regresyon modeli $R^2 = 0.4284$ seviyesinde kalırken, ağaç tabanlı hiperparametre optimizasyonlu modeller hata oranını yarı yarıya düşürmüştür.
3. **Fiyatı En Çok Etkileyen Değişkenler:**
   - `Duration_mins` (Uçuş Süresi)
   - `Is_Premium_Airline` (Vistara / Air India vs IndiGo / SpiceJet / Air Asia)
   - `Lead_Time_Days` (Erken Rezervasyon Günü)
   - `Stops_Num` (Aktarma Sayısı)

---

## 📂 Proje Yapısı

```
.
├── flight_analysis_and_modeling.ipynb  # Ana Jupyter Notebook (EDA + Features + ML Models + Hyperparameter Tuning)
├── flight_cleaned.csv                  # Orijinal uçuş verisi
├── flight_processed.csv                # Temizlenmiş ve işlenmiş veri seti
├── build_notebook.py                   # Notebook'u oluşturan otomasyon kodu
├── eda_analysis.py                     # EDA betiği
├── eda_summary_charts.png              # EDA özet grafik görseli
└── README.md                           # Proje dokümantasyonu
```
