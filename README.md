# ✈️ Uçuş Bilet Fiyatı Analizi ve Makine Öğrenmesi Tahmin Modeli (VODP)

Bu proje, Hindistan içi iç hat uçuş veri seti (`flight_cleaned.csv`) üzerinde gerçekleştirilen **Keşifçi Veri Analizi (EDA)**, **Özellik Mühendisliği (Feature Engineering)** ve **Çeşitli Regresyon Modelleri (Linear, Tree-based & Ensemble ML)** ile uçuş bilet fiyatı tahminleme çalışmasını içermektedir.

---

## 📊 Makine Öğrenmesi Model Başarı Karşılaştırması

Tüm regresyon modelleri **%80 Train / %20 Test** ayrımı ve ön işleme pipeline'ı (`StandardScaler` + `OneHotEncoder`) ile değerlendirilmiştir:

| Model | $R^2$ Score (Açıklanan Varyans) | MAE (Ortalama Mutlak Hata) | RMSE (Kök Ortalama Kare Hata) |
| :--- | :---: | :---: | :---: |
| 🥇 **Random Forest Regressor** | **0.7347** | **1,196.95 ₹** | **2,179.68 ₹** |
| 🥈 **LightGBM Regressor** | **0.7296** | **1,311.34 ₹** | **2,200.48 ₹** |
| 🥉 **XGBoost Regressor** | **0.7273** | **1,278.95 ₹** | **2,209.80 ₹** |
| 🔹 **Gradient Boosting** | **0.6494** | **1,612.46 ₹** | **2,505.72 ₹** |
| 🔹 **Decision Tree Regressor** | **0.4856** | **1,467.32 ₹** | **3,035.24 ₹** |
| 🔸 **Lasso Regression** | **0.4289** | **2,282.13 ₹** | **3,197.95 ₹** |
| 🔸 **Ridge Regression** | **0.4288** | **2,281.58 ₹** | **3,198.28 ₹** |
| 🔸 **Linear Regression (Baseline)** | **0.4284** | **2,283.70 ₹** | **3,199.53 ₹** |

---

## 🔑 Önemli Çıkarımlar & Bulgular

1. **Topluluk Modellerinin Üstünlüğü:** Ağaç tabanlı topluluk modelleri (**Random Forest**, **LightGBM**, **XGBoost**), bilet fiyatlarındaki doğrusal olmayan ilişkileri başarıyla yakalayarak $R^2$ değerini **%73.5** seviyesine çıkarmış ve ortalama hatayı (MAE) **1,196 ₹** seviyesine düşürmüştür.
2. **Fiyatı En Çok Etkileyen Değişkenler:**
   - `Duration_mins` (Uçuş Süresi)
   - `Is_Premium_Airline` (Vistara / Air India vs IndiGo / SpiceJet / Air Asia)
   - `Lead_Time_Days` (Erken Rezervasyon Günü)
   - `Stops_Num` (Aktarma Sayısı)

---

## 📂 Proje Yapısı

```
.
├── flight_analysis_and_modeling.ipynb  # Ana Jupyter Notebook (EDA + Feature Engineering + ML Models)
├── flight_cleaned.csv                  # Orijinal uçuş verisi
├── flight_processed.csv                # Temizlenmiş ve işlenmiş veri seti
├── build_notebook.py                   # Notebook'u programatik oluşturan Python betiği
├── eda_analysis.py                     # EDA görselleştirme betiği
├── eda_summary_charts.png              # EDA özet grafik görseli
├── process_india_airports.py           # Hindistan havalimanı işleme kodu
├── airports_india.csv                  # Filtrelenmiş Hindistan havalimanları
└── README.md                           # Proje dokümantasyonu
```

---

## 🚀 Çalıştırma

Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm jupyter
```

Jupyter Notebook'u başlatın:
```bash
jupyter notebook flight_analysis_and_modeling.ipynb
```
