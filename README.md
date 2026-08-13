# Flight Data Analysis (VODP)

Bu proje, Hindistan uçuş verileri (`flight_cleaned.csv`) ve küresel havalimanı verileri (`airports.csv`) üzerinde gerçekleştirilen veri temizleme, keşifçi veri analizi (EDA) ve görselleştirme çalışmalarını içermektedir.

## 📂 Proje İçeriği

- **`eda_analysis.py`**: Veri temizleme, istatistiksel özetleme ve grafik üretimi yapan ana Python betiği.
- **`flight_cleaned.csv`**: Orijinal uçuş veri seti.
- **`flight_processed.csv`**: Temizlenmiş ve standartlaştırılmış uçuş veri seti.
- **`airports.csv`**: Küresel havalimanı ve koordinat verileri.
- **`eda_summary_charts.png`**: Havayolu fiyat dağılımları, erken rezervasyon analizi, rota fiyatları ve uçuş sürelerini gösteren grafikler.

## 🚀 Çalıştırma

Gerekli kütüphaneleri yükleyin:
```bash
pip install pandas numpy matplotlib seaborn
```

Analiz betiğini çalıştırın:
```bash
python eda_analysis.py
```
