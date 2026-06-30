import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

model = joblib.load('models/aussie_rain.joblib')

locations = model['encoder'].categories_[0]
wind_dirs = model['encoder'].categories_[1]


def optional_select(label, options, key):
    """Selectbox з опцією 'Невідомо' (→ NaN)."""
    clean_options = [opt for opt in options if pd.notna(opt)]
    choice = st.selectbox(label, clean_options + ["— Невідомо —"], key=key)
    return None if choice == "— Невідомо —" else choice
 
def optional_number(label, min_v, max_v, step, fmt, key, help=None):
    """Числове поле з чекбоксом «Невідомо»."""
    unknown = st.checkbox("Невідомо", key=f"{key}_unk", help="Залиш, якщо даних немає")
    if unknown:
        return None
    return st.number_input(label, min_value=min_v, max_value=max_v,
                           step=step, format=fmt, key=key, help=help)
 

st.set_page_config(
    page_title="Rain in Australia — Prediction",
    page_icon="🌧️",
    layout="wide",
)
 
st.title("🌧️ Прогноз дощу завтра")
st.caption("Заповніть погодні дані за сьогодні — модель передбачить, чи буде дощ завтра.")


# SECTION 1 - location
st.subheader("📍 Локація")
location = st.selectbox("Місто / станція спостереження", locations, key="location")


# SECTION 2 — Temperature & Rain
st.subheader("🌡️ Температура та опади")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("Мін. температура (°C)")
    min_temp = optional_number("MinTemp", -10.0, 50.0, 0.1, "%.1f", "min_temp")

with col2:
    st.markdown("Макс. температура (°C)")
    max_temp = optional_number("MaxTemp", -10.0, 55.0, 0.1, "%.1f", "max_temp")

with col3:
    st.markdown("Опади (мм)")
    rainfall = optional_number("Rainfall", 0.0, 400.0, 0.1, "%.1f", "rainfall",
                               help="Сума опадів за добу")
    

# SECTION 3 — Evaporation & Sunshine
st.subheader("Випаровування та сонце")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Випаровування (мм)")
    evap = optional_number("Evaporation", 0.0, 150.0, 0.1, "%.1f", "evaporation")

with col2:
    st.markdown("Сонячні години")
    sunsh = optional_number("Sunshine", 0.0, 14.5, 0.1, "%.1f", "sunshine")


# SECTION 4 — Wind
st.subheader("💨 Вітер")
col1, col2 = st.columns(2)
 
with col1:
    wind_gust_dir = optional_select("Напрям поривів (WindGustDir)", wind_dirs, "wgd")
    st.markdown("**Швидкість поривів (км/год)**")
    wind_gust_speed = optional_number("WindGustSpeed", 0.0, 200.0, 1.0, "%.0f", "wgs")
 
with col2:
    col_a, col_b = st.columns(2)
    with col_a:
        wind_dir_9am = optional_select("Напрям о 9:00", wind_dirs, "wd9")
        st.markdown("**Швидкість о 9:00 (км/год)**")
        wind_speed_9am = optional_number("WindSpeed9am", 0.0, 150.0, 1.0, "%.0f", "ws9")
    with col_b:
        wind_dir_3pm = optional_select("Напрям о 15:00", wind_dirs, "wd3")
        st.markdown("**Швидкість о 15:00 (км/год)**")
        wind_speed_3pm = optional_number("WindSpeed3pm", 0.0, 150.0, 1.0, "%.0f", "ws3")
 

# SECTION 5 — Humidity & Pressurest.subheader("💧 Вологість та тиск")
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("**Вологість о 9:00 (%)**")
    humidity_9am = optional_number("Humidity9am", 0, 100, 1, "%d", "h9")
    st.markdown("**Тиск о 9:00 (гПа)**")
    pressure_9am = optional_number("Pressure9am", 900.0, 1100.0, 0.1, "%.1f", "p9")
 
with col2:
    st.markdown("**Вологість о 15:00 (%)**")
    humidity_3pm = optional_number("Humidity3pm", 0, 100, 1, "%d", "h3")
    st.markdown("**Тиск о 15:00 (гПа)**")
    pressure_3pm = optional_number("Pressure3pm", 900.0, 1100.0, 0.1, "%.1f", "p3")
 

# SECTION 6 — Cloud & Temp snapshots
st.subheader("☁️ Хмарність та температура вдень")
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("**Хмарність о 9:00 (октанти 0–9)**")
    cloud_9am = optional_number("Cloud9am", 0, 9, 1, "%d", "c9",
                                help="0 = ясно, 8 = повністю хмарно")
    st.markdown("**Температура о 9:00 (°C)**")
    temp_9am = optional_number("Temp9am", -10.0, 50.0, 0.1, "%.1f", "t9")
 
with col2:
    st.markdown("**Хмарність о 15:00 (октанти 0–9)**")
    cloud_3pm = optional_number("Cloud3pm", 0, 9, 1, "%d", "c3")
    st.markdown("**Температура о 15:00 (°C)**")
    temp_3pm = optional_number("Temp3pm", -10.0, 50.0, 0.1, "%.1f", "t3")


# SECTION 7 — Rain today
st.subheader("🌂 Сьогодні йшов дощ?")
rain_today = st.radio(
    "RainToday",
    options=["Yes", "No"],
    format_func=lambda x: "✅ Так" if x == "Yes" else "❌ Ні",
    horizontal=True,
    key="rain_today",
)

# PREDICTION
if st.button("🔮 Передбачити", type="primary", use_container_width=True):
    data_dict ={
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evap,
        "Sunshine": sunsh,
        "WindGustDir":   wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am":    wind_dir_9am,
        "WindDir3pm":    wind_dir_3pm,
        "WindSpeed9am":  wind_speed_9am,
        "WindSpeed3pm":  wind_speed_3pm,
        "Humidity9am":   humidity_9am,
        "Humidity3pm":   humidity_3pm,
        "Pressure9am":   pressure_9am,
        "Pressure3pm":   pressure_3pm,
        "Cloud9am":      cloud_9am,
        "Cloud3pm":      cloud_3pm,
        "Temp9am":       temp_9am,
        "Temp3pm":       temp_3pm,
        "RainToday":     rain_today,
    }
    
    df = pd.DataFrame([data_dict])
    
    num_cols = model['numeric_cols']
    cat_cols = model['categorical_cols']
    
    ## Transform data
    df[num_cols] = model['imputer'].transform(df[num_cols])
    df[num_cols] = model['scaler'].transform(df[num_cols])
    
    cat_arr = model['encoder'].transform(df[cat_cols])
    
    df_encoded = pd.DataFrame(cat_arr, columns=model['encoded_cols'])
    
    df_final = pd.concat(
            [df[num_cols].reset_index(drop=True), df_encoded],
            axis=1,
        )
    
    prediction = model['model'].predict(df_final)[0]
    pred_proba = model['model'].predict_proba(df_final)[0]
    
    if prediction == "Yes":
            st.success("🌧️ **Прогноз: завтра очікується дощ**")
    else:
        st.info("☀️ **Прогноз: завтра дощу не очікується**")
    
    prob_df = pd.DataFrame({
                "Клас": [str(c) for c in model['model'].classes_],
                "Ймовірність": pred_proba,
            })
    st.bar_chart(prob_df.set_index("Клас"))
     
    with st.expander("🔍 Переглянути вхідні дані"):
        st.dataframe(df, use_container_width=True)