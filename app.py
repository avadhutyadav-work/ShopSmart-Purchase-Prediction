import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("ShopSmart - Purchase Prediction")
st.write("Enter the session details to predict if the visitor will buy.")

# Same numbers that LabelEncoder gave in the notebook (alphabetical order)
month_map = {"Aug":0, "Dec":1, "Feb":2, "Jul":3, "June":4,
             "Mar":5, "May":6, "Nov":7, "Oct":8, "Sep":9}
visitor_map = {"New_Visitor":0, "Other":1, "Returning_Visitor":2}

col1, col2 = st.columns(2)
with col1:
    administrative = st.number_input("Administrative pages", 0, 50, 0)
    administrative_duration = st.number_input("Administrative duration", 0.0, 5000.0, 0.0)
    informational = st.number_input("Informational pages", 0, 30, 0)
    informational_duration = st.number_input("Informational duration", 0.0, 5000.0, 0.0)
    product_related = st.number_input("Product related pages", 0, 800, 10)
    product_duration = st.number_input("Product related duration", 0.0, 70000.0, 300.0)
    bounce = st.number_input("Bounce rate", 0.0, 0.2, 0.02, format="%.4f")
    exit_rate = st.number_input("Exit rate", 0.0, 0.2, 0.04, format="%.4f")
    page_values = st.number_input("Page values", 0.0, 400.0, 0.0)
with col2:
    special_day = st.selectbox("Special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    month = st.selectbox("Month", list(month_map.keys()))
    os_ = st.number_input("Operating system (id)", 1, 8, 2)
    browser = st.number_input("Browser (id)", 1, 13, 2)
    region = st.number_input("Region (id)", 1, 9, 1)
    traffic = st.number_input("Traffic type (id)", 1, 20, 2)
    visitor = st.selectbox("Visitor type", list(visitor_map.keys()))
    weekend = st.selectbox("Weekend", ["No", "Yes"])

if st.button("Predict"):
    row = {
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month_map[month],
        "OperatingSystems": os_,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic,
        "VisitorType": visitor_map[visitor],
        "Weekend": 1 if weekend == "Yes" else 0,
    }
    df = pd.DataFrame([row])[columns]          # same column order as training
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.success(f"Likely to BUY (probability {prob:.0%})")
    else:
        st.warning(f"Unlikely to buy (probability of buying {prob:.0%})")