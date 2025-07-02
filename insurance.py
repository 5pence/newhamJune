import streamlit as st   
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Healthcare Cost Estimator", layout="wide")


# load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.title("Healthcare Insurance Cost Analysis & Estimator")

# Sidebar filers
st.sidebar.header("Filter data")
selected_region = st.siderbar.multiselect(
    "Select Region(s):",
    options=df['region'].unique(),
    default=df['region'].unique()
)
selected_smoker = st.sidebar.selectbox(
    "Smoker:",
    options=["both", "yes", "no"]
)

# Filter the DataFrame
filtered_df = df[df['region'].isin(selected_region)]
if selected_smoker != "both":
    filtered_df = filtered_df[filtered_df['smoker'] == selected_smoker]


# BMI Categories
def bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Healthy"
    elif bmi < 30: return "Overweight"
    else: return "Obese"


filtered_df['bmi_category'] = filered_df['bmi'].apply(bmi_category)

# Tabs
tab1, tab2, tab3 = st.tabs(["Visual insights", "Cost Prediction", "Raw Data"])

with tab1:
    st.header("Average Charegs by Demographic")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Charges by smoker status")
        sns.boxplot(data=filtered_df, x="smoker", y="charges")
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.subheader("Charge by BMI Category")
        sns.boxplot(data=filtered_df, x="bmi_category", y="charges")
        st.pyplot(plt.gcf())
        plt.clf()

    st.subheader("Correlation Heatmap")
    corr = filtered_df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

with tab2:
    st.header("Estimate your insurance Cost")

    # Encode categorical features using seperate encoders
    sex_encoder = LabelEncoder()
    smoker_encoder = LabelEncoder()
    region_encoder = LabelEncoder()

    df['sex_enc'] = sex_encoder.fit_transform(df['sex'])
    df['smoker_enc'] = smoker_encoder.fit_transform(df['smoker'])
    df['region_enc'] = region_encoder.fit_transform(df['region'])

    model_df = df[['age', 'sex_enc', 'bmi', 'children', 'smoker_enc', 'region_enc', 'charges']]
    X = model_df.drop('charges', axis=1)
    y = model_df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Input form
    with st.form("predict_form"):
        age = st.slider("Age", 18, 64, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.slider("BMI", 15.0, 40.0, 25.0)
        children = st.slider("Number of Children", 0, 5, 1)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", df["region"].unique())

        submitted = st.form_submit_button("Predict Cost")

        if submitted:
            input_data = pd.DataFrame([{
                "age": age,
                "sex_enc": sex_encoder.transform([sex])[0],
                "bmi": bmi,
                "children": children,
                "smoker_enc": smoker_encoder.transform([smoker])[0],
                "region_enc": region_encoder.transform([region])[0]
            }])

            predicted_cost = model.predict(input_data)[0]
            st.success(f"💸 Estimated Insurance Charge: ${predicted_cost:,.2f}")

with tab3:
    st.header("📎 Raw Filtered Data")
    st.dataframe(filtered_df)
