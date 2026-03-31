import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("titanic.csv")

st.title("Summary of the Titanic tragedy")

st.header("Dataset Overview")
st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.subheader("Column Types")
st.write(df.dtypes)

st.subheader("Descriptive Statistics")
st.write(df.describe(include='all'))

st.header("Preview Data")
n_rows = st.slider("Select number of rows to display", 1, 100, 5)
st.dataframe(df.head(n_rows))

st.header("Visualizations")

# Survival count
fig1 = px.histogram(df, x="Survived", title="Survival count", color="Survived",
                    color_discrete_map={0: "red", 1: "green"}, opacity=0.6)
fig1.update_traces(marker_line_color='white', marker_line_width=2)
fig1.update_layout(bargap=0.2, xaxis_title="Passenger status", yaxis_title="Count", showlegend=False)
fig1.update_xaxes(tickmode = 'array', tickvals = [0, 1], ticktext = ['Dead', 'Survived'])
st.plotly_chart(fig1, use_container_width=True)

# Age distribution
fig2 = px.violin(df, x="Survived", y="Age", title="Age distribution", color="Survived",
                 color_discrete_map={0: "red", 1: "green"})
fig2.update_layout(xaxis_title="Passenger status", yaxis_title="Age", showlegend=False)
fig2.update_xaxes(tickmode = 'array', tickvals = [0, 1], ticktext = ['Dead', 'Survived'])
st.plotly_chart(fig2, use_container_width=True)

# Fare distribution by passenger class
fig3 = px.box(df, x="Pclass", y="Fare", title="Fare distribution by passenger class", color="Pclass")
fig3.update_layout(xaxis_title="Passenger class", yaxis_title="Fare", showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# Survival count by gender
fig4 = px.histogram(df, x="Sex", color="Survived", barmode="group",
                    title="Survival count by gender", color_discrete_map={0: "red", 1: "green"}, opacity=0.6)
fig4.update_traces(marker_line_color='white', marker_line_width=2)
fig4.update_layout(bargap=0.2, xaxis_title="Sex", yaxis_title="Count")
st.plotly_chart(fig4, use_container_width=True)

# Interactive plot
st.subheader("Another plots")
selected_feature = st.selectbox("Select feature to analyze against survival", ["SibSp", "Parch"])
fig5 = px.histogram(df, x=selected_feature, color="Survived", barmode="group",
                    title=f"Survival by {selected_feature}", color_discrete_map={0: "red", 1: "green"},
                    opacity=0.6)
fig5.update_traces(marker_line_color='white', marker_line_width=2)
st.plotly_chart(fig5, use_container_width=True)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include='number')
fig6 = px.imshow(numeric_df.corr(), text_auto=True, title="Correlation Matrix", color_continuous_scale='RdBu',
                 range_color=[-1, 1], color_continuous_midpoint=0)
st.plotly_chart(fig6, use_container_width=True)
