import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime

# Page Configuration
st.set_page_config(
    page_title="Uçuş Bilet Fiyatı Tahmini",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #2563EB;
    }
    .price-range {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 500;
    }
    .badge-premium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-budget {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Metadata
@st.cache_resource
def load_model_and_metadata():
    model = joblib.load('flight_model.joblib')
    with open('flight_metadata.json', 'r') as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_model_and_metadata()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# Header
st.markdown('<div class="main-header">✈️ Uçuş Bilet Fiyatı Tahminleme Portalı</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Gelişmiş Makine Öğrenmesi (LightGBM Tuned Model - R²: %75.7) ile bilet fiyatı tahmini.</div>', unsafe_allow_html=True)

# Layout: 2 Columns (Form on Left, Predictions & Analytics on Right)
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📋 Uçuş ve Rezervasyon Parametreleri")
    
    # 1. Havayolu Seçimi
    airline = st.selectbox(
        "Havayolu Şirketi",
        options=meta['airlines'],
        index=meta['airlines'].index("IndiGo") if "IndiGo" in meta['airlines'] else 0,
        help="Premium: Vistara, Air India | Bütçe: IndiGo, SpiceJet, Air Asia, GO FIRST"
    )
    
    is_premium = 1 if airline in ['Vistara', 'Air India'] else 0
    if is_premium:
        st.caption("✨ **Premium Segment:** Yemek, bagaj ve tam hizmet dahildir.")
    else:
        st.caption("💡 **Bütçe Segmenti:** Standart ekonomi bütçe uçuşu.")
        
    st.divider()
    
    # 2. Rota Seçimi (From -> To)
    c1, c2 = st.columns(2)
    with c1:
        from_city = st.selectbox("Kalkış Şehri (From)", options=meta['cities'], index=meta['cities'].index("Delhi") if "Delhi" in meta['cities'] else 0)
    with c2:
        valid_to_cities = [c for c in meta['cities'] if c != from_city]
        to_city = st.selectbox("Varış Şehri (To)", options=valid_to_cities, index=valid_to_cities.index("Bangalore") if "Bangalore" in valid_to_cities else 0)
        
    route = f"{from_city} -> {to_city}"
    
    st.divider()
    
    # 3. Tarih Seçimleri & Lead Time
    c3, c4 = st.columns(2)
    with c3:
        booking_date = st.date_input("Rezervasyon Tarihi", value=datetime.date(2022, 6, 23))
    with c4:
        journey_date = st.date_input("Uçuş Tarihi", value=datetime.date(2022, 6, 26), min_value=booking_date)
        
    lead_time_days = (journey_date - booking_date).days
    
    if lead_time_days == 0:
        st.warning("⚠️ **Son Dakika Rezervasyonu:** Uçuş bugün yapılıyor. Fiyat yükselme riski yüksektir.")
    else:
        st.info(f"📅 **Erken Rezervasyon:** Uçuştan **{lead_time_days} gün önce** alınıyor.")
        
    st.divider()
    
    # 4. Kalkış Saati, Uçuş Süresi & Aktarma
    c5, c6 = st.columns(2)
    with c5:
        dep_time = st.time_input("Kalkış Saati", value=datetime.time(14, 30))
        stops_str = st.selectbox("Aktarma Sayısı", options=["Direkt Uçuş (non-stop)", "1 Aktarma (1 stop)", "2 Aktarma (2 stops)"], index=0)
    with c6:
        duration_hours = st.number_input("Uçuş Süresi (Saat)", min_value=1, max_value=24, value=2)
        duration_mins_part = st.number_input("Uçuş Süresi (Dakika)", min_value=0, max_value=59, value=45)
        
    duration_mins = (duration_hours * 60) + duration_mins_part

# Derived Input Values
stops_num = 0 if "non-stop" in stops_str else (1 if "1 Aktarma" in stops_str else 2)

dep_hour = dep_time.hour
dep_minute = dep_time.minute

def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Night'

dep_time_of_day = get_time_of_day(dep_hour)

# Estimate Arrival Time & Next Day Flag
arr_hour = (dep_hour + duration_hours + (dep_minute + duration_mins_part) // 60) % 24
arr_minute = (dep_minute + duration_mins_part) % 60
is_next_day = 1 if (dep_hour + duration_hours) >= 24 else 0
arr_time_of_day = get_time_of_day(arr_hour)

journey_day = journey_date.day
journey_month = journey_date.month
journey_dayofweek = journey_date.weekday()
journey_is_weekend = 1 if journey_dayofweek in [5, 6] else 0

booking_hour = 12  # Standard default booking hour
booking_slot = 'Afternoon'

# Build Feature Input Dataframe
input_dict = {
    'Airline': airline,
    'From': from_city,
    'To': to_city,
    'Route': route,
    'Booking_Slot': booking_slot,
    'Dep_TimeOfDay': dep_time_of_day,
    'Arr_TimeOfDay': arr_time_of_day,
    'Stops_Num': stops_num,
    'Booking_Hour': booking_hour,
    'Lead_Time_Days': lead_time_days,
    'Duration_mins': duration_mins,
    'Journey_Day': journey_day,
    'Journey_DayOfWeek': journey_dayofweek,
    'Journey_IsWeekend': journey_is_weekend,
    'Dep_Hour': dep_hour,
    'Dep_Minute': dep_minute,
    'Arr_Hour': arr_hour,
    'Arr_Minute': arr_minute,
    'Is_Next_Day': is_next_day,
    'Is_Premium_Airline': is_premium
}

input_df = pd.DataFrame([input_dict])

with col_right:
    st.subheader("💡 Tahmin ve Analiz Sonuçları")
    
    # Predict button / Auto Predict
    pred_price = model.predict(input_df)[0]
    
    # Format Price Output
    formatted_price = f"₹ {pred_price:,.0f}"
    
    # Confidence Interval based on MAE (~1,200 INR)
    mae_margin = 1200
    min_est = max(7000, pred_price - mae_margin)
    max_est = pred_price + mae_margin
    
    # Result Card Container
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;">Tahmin Edilen Bilet Fiyatı</div>
        <div class="price-value">{formatted_price}</div>
        <div class="price-range">Tahmini Güven Aralığı: <b>₹ {min_est:,.0f} - ₹ {max_est:,.0f}</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Summary Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Güzargah", route)
    with m2:
        st.metric("Erken Rezervasyon", f"{lead_time_days} Gün Önce")
    with m3:
        st.metric("Uçuş Süresi", f"{duration_hours}s {duration_mins_part}dk")
        
    st.divider()
    
    # Recommendation Box
    st.subheader("📌 Model Tavsiyesi & Fiyat İçgörüsü")
    
    if lead_time_days >= 3:
        st.success("✅ **Erken Rezervasyon Avantajı:** Uçuştan birkaç gün önce rezervasyon yaptığınız için fiyatlar daha kararlı seviyededir.")
    else:
        st.warning("⚡ **Yaklaşan Uçuş:** Uçuş gününe çok yakın rezervasyon yapılıyor. Havayolu doluluğuna bağlı fiyat artışları görülebilir.")
        
    if stops_num > 0:
        st.info("ℹ️ **Aktarmalı Uçuş:** Aktarmalı uçuşlar operasyonel maliyetlerden dolayı direkt uçuşlara kıyasla farklı fiyatlandırılır.")
        
    with st.expander("🔍 Girdi Değişkenleri Detayını İncele"):
        st.json(input_dict)

st.divider()
st.caption("🚀 **Vodptk Flight Predictor** | LightGBM Machine Learning Model Pipeline | Python 3.13 & Streamlit")
