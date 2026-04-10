import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import *

st.set_page_config(page_title="ShopIntent AI", page_icon="🛒", layout="wide")

# ---------------- DATA ---------------- #
@st.cache_data
def generate_dataset(n=10000):
    np.random.seed(42)
    df = pd.DataFrame({
        'PageValues': np.random.exponential(6, n),
        'BounceRates': np.random.beta(1.5, 20, n),
        'ExitRates': np.random.beta(2, 15, n),
        'ProductRelated_Duration': np.random.exponential(1200, n),
        'VisitorType': np.random.choice(['Returning_Visitor','New_Visitor','Other'], n),
        'Weekend': np.random.choice([0,1], n)
    })

    logit = -2.5 + 0.05*df.PageValues -3*df.BounceRates -1.5*df.ExitRates + 0.01*df.ProductRelated_Duration/100
    prob = 1/(1+np.exp(-logit))
    df['Revenue'] = np.random.binomial(1, prob)

    return df

df = generate_dataset()

# ---------------- PREPROCESS ---------------- #
def preprocess(df):
    df = df.copy()
    df['VisitorType'] = LabelEncoder().fit_transform(df['VisitorType'])
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    return X, y

# ---------------- MODELS ---------------- #
MODELS = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=500)
}

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
train_btn = st.sidebar.button("🚀 Train Model")

# ---------------- TABS ---------------- #
tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Train", "🔮 Predict"])

# ---------------- TAB 1 ---------------- #
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    df['Revenue'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)

# ---------------- TAB 2 ---------------- #
with tab2:
    if train_btn:
        X, y = preprocess(df)

        scaler = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = MODELS[model_name]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        # SAVE MODEL + SCALER 🔥
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.columns = X.columns

        st.success("Model Trained Successfully ✅")

        st.write("### Metrics")
        st.write({
            "Accuracy": accuracy_score(y_test,y_pred),
            "Precision": precision_score(y_test,y_pred),
            "Recall": recall_score(y_test,y_pred),
            "F1 Score": f1_score(y_test,y_pred),
            "ROC AUC": roc_auc_score(y_test,y_prob)
        })

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

# ---------------- TAB 3 ---------------- #
with tab3:
    if "model" not in st.session_state:
        st.warning("⚠️ Train model first")
    else:
        st.subheader("Live Prediction")

        pv = st.slider("Page Value", 0.0, 100.0, 10.0)
        br = st.slider("Bounce Rate", 0.0, 1.0, 0.1)
        er = st.slider("Exit Rate", 0.0, 1.0, 0.1)
        pdur = st.slider("Product Duration", 0, 5000, 1000)
        vt = st.selectbox("Visitor Type", ['Returning_Visitor','New_Visitor','Other'])
        wk = st.checkbox("Weekend")

        if st.button("Predict"):
            # ENCODE VisitorType 🔥
            visitor_map = {'Other': 0, 'Returning_Visitor': 1, 'New_Visitor': 2}
            vt = visitor_map.get(vt, 0)

            # CREATE INPUT DATAFRAME
            input_df = pd.DataFrame([{
                'PageValues': pv,
                'BounceRates': br,
                'ExitRates': er,
                'ProductRelated_Duration': pdur,
                'VisitorType': vt,
                'Weekend': int(wk)
            }])

            # SCALE USING TRAINED SCALER
            input_scaled = st.session_state.scaler.transform(input_df)

            model = st.session_state.model
            prob = model.predict_proba(input_scaled)[0][1]

            st.success(f"🛒 Purchase Probability: {prob*100:.2f}%")