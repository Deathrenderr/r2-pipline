import streamlit as st
import pandas as pd

from pipeline import load_data, clean_data, train_model

st.title("Datanyx Round 2 - ML Demo App")

file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.write(df.head())

    df = clean_data(df)

    target = st.selectbox("Select target column", df.columns)
    mode = st.selectbox("Model Type", ["classification", "regression"])

    if st.button("Train Model"):
        model = train_model(df, target, mode)
        st.success("Model trained successfully!")
        st.write("Model object:", model)
