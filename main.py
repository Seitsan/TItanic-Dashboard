import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------
# Load data
# ----------------------
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

st.title("🚢 Titanic Dashboard")

# ----------------------
# Basic Info
# ----------------------
st.header("📊 Dataset Overview")

st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.subheader("Column Types")
st.write(df.dtypes)

st.subheader("Descriptive Statistics")
st.write(df.describe(include='all'))

# ----------------------
# Show N rows
# ----------------------
st.header("🔎 Preview Data")

n_rows = st.slider("Select number of rows to display", 1, 100, 5)
st.dataframe(df.head(n_rows))

# ----------------------
# Visualizations
# ----------------------
st.header("📈 Visualizations")

# 1. Survival count
fig1 = px.histogram(df, x="Survived", title="Survival Count")
st.plotly_chart(fig1, use_container_width=True)

# 2. Age distribution
fig2 = px.histogram(df, x="Age", nbins=30, title="Age Distribution")
st.plotly_chart(fig2, use_container_width=True)

# 3. Fare distribution by class
fig3 = px.box(df, x="Pclass", y="Fare", title="Fare by Passenger Class")
st.plotly_chart(fig3, use_container_width=True)

# 4. Gender vs Survival
fig4 = px.histogram(df, x="Sex", color="Survived", barmode="group",
                    title="Survival by Gender")
st.plotly_chart(fig4, use_container_width=True)

# 5. Interactive plot (user input)
st.subheader("🎯 Interactive Plot")

selected_feature = st.selectbox(
    "Select feature to analyze against survival",
    ["Pclass", "Sex", "SibSp", "Parch"]
)

fig5 = px.histogram(
    df,
    x=selected_feature,
    color="Survived",
    barmode="group",
    title=f"Survival vs {selected_feature}"
)

st.plotly_chart(fig5, use_container_width=True)

# ----------------------
# Correlation Heatmap
# ----------------------
st.subheader("🔥 Correlation Heatmap")

numeric_df = df.select_dtypes(include='number')

fig6 = px.imshow(numeric_df.corr(), text_auto=True, title="Correlation Matrix")
st.plotly_chart(fig6, use_container_width=True)
