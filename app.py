import streamlit as st
import pandas as pd
import joblib

# Load models and data
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_rfm.pkl")
product_sim_df = pd.read_pickle("product_similarity.pkl")

st.set_page_config(page_title="🛒 Shopper Spectrum", layout="centered")

st.title("🛍️ Shopper Spectrum")
st.markdown("A smart e-commerce assistant for **customer segmentation** and **product recommendations**.")

tab1, tab2 = st.tabs(["🔍 Product Recommendations", "👥 Customer Segmentation"])

# ---------------------------- PRODUCT RECOMMENDATION ---------------------------- #
with tab1:
    st.subheader("🔁 Get Similar Products")
    product_name = st.text_input("Enter a product name (case-sensitive):")

    if st.button("🧠 Recommend Similar Products"):
        if product_name in product_sim_df.columns:
            sim_products = product_sim_df[product_name].sort_values(ascending=False)[1:6]
            st.success("Top 5 Recommended Products:")
            for i, product in enumerate(sim_products.index, 1):
                st.markdown(f"**{i}. {product}**")
        else:
            st.error("⚠️ Product not found. Try another name from your dataset.")

# ---------------------------- CUSTOMER SEGMENTATION ---------------------------- #
with tab2:
    st.subheader("📊 Predict Customer Segment")
    r = st.number_input("📅 Recency (days ago)", min_value=0, value=30)
    f = st.number_input("🔁 Frequency (number of transactions)", min_value=0, value=5)
    m = st.number_input("💰 Monetary (total spend)", min_value=0.0, value=100.0)

    if st.button("🎯 Predict Segment"):
        user_data = scaler.transform([[r, f, m]])
        cluster = kmeans.predict(user_data)[0]

        # Segment mapping logic
        def map_segment(cluster_id):
            if cluster_id == 0:
                return "Occasional"
            elif cluster_id == 1:
                return "At-Risk"
            elif cluster_id == 2:
                return "Regular"
            elif cluster_id == 3:
                return "High-Value"
            else:
                return "Unknown"

        segment = map_segment(cluster)
        st.success(f"🧩 Predicted Customer Segment: **{segment}**")
