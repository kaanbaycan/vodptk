# ✈️ Uçuş Bilet Fiyatı Tahminleme Web Portalı (VODP)

Bu proje, Hindistan içi iç hat uçuş veri seti (`flight_cleaned.csv`) üzerinde gerçekleştirilen **Keşifçi Veri Analizi (EDA)**, **Özellik Mühendisliği (Feature Engineering)**, **Topluluk Regresyon Modelleri**, **Hiperparametre Optimizasyonu (RandomizedSearchCV)** ve **Streamlit Web Uygulaması** çalışmalarını kapsamaktadır.

---

## 🚀 Streamlit Uygulamasını Yerel Olarak Çalıştırma

Streamlit arayüzünü başlatmak için şu komutu çalıştırın:

```bash
cd /Users/yakupkaanbaycan/vodp
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılan ekranda (`http://localhost:8501`) kalkış şehri, varış şehri, havayolu şirketi, uçuş tarihi ve kalkış saati girerek **anlık fiyat tahmini** alabilirsiniz.

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

## 📂 Proje Yapısı

```
.
├── app.py                              # Streamlit Web Uygulaması (Canlı Fiyat Tahmin Arayüzü)
├── flight_model.joblib                 # Eğitilmiş ve dışa aktarılmış LightGBM Model Pipeline'ı
├── flight_metadata.json                # Arayüz için şehir, havayolu ve meta verileri
├── train_and_save_model.py             # Modeli eğitip joblib olarak kaydeden betik
├── flight_analysis_and_modeling.ipynb  # Ana Jupyter Notebook (EDA + Features + ML Models + Tuning + Hata Analizi)
├── flight_cleaned.csv                  # Orijinal uçuş verisi
├── flight_processed.csv                # Temizlenmiş ve işlenmiş veri seti
├── build_notebook.py                   # Notebook otomasyon kodu
├── eda_analysis.py                     # EDA betiği
└── README.md                           # Proje dokümantasyonu
```
