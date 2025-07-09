import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Load model and data
filename = 'random_forest_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("New_Clustered_Customer_Data.csv")

# Set page configuration
st.set_page_config(page_title="Customer Predictions", layout="wide")

# Load the background image
bg_image = Image.open("Background_img.jpg") 

# Add background image
st.image(bg_image, use_container_width=True, caption='', clamp=True) 

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        color: white;
        background-color: #282c34;
        font-family: 'Helvetica', sans-serif;
    }
    .title {
            color: black;
            font-size: 5em;
            }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 10px;
    }
    .stNumberInput > div > input, .stTextInput > div > input, .stSelectbox > div > select, .stCheckbox > div > input {
        background-color: rgba(255, 255, 255, 0.8);
        color: #333;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
    }
    .main-title {
        text-align: center;
        color: white;
        font-size: 3em;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="main-title">Customer Segmentation Predictions</h1>', unsafe_allow_html=True)

# Sidebar for input features
st.sidebar.header("Enter Inputs")
with st.sidebar.form("my_form"):
    balance = st.number_input(label='Balance', step=0.001, format="%.6f")
    balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
    purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
    oneoff_purchases = st.number_input(label='One-Off Purchases', step=0.01, format="%.2f")
    installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
    cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
    purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input(label='One-Off Purchases Frequency', step=0.1, format="%.6f")
    purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', step=0.1, format="%.6f")
    cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
    cash_advance_trx = st.number_input(label='Cash Advance Transactions', step=1)
    purchases_trx = st.number_input(label='Purchases Transactions', step=1)
    credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
    payments = st.number_input(label='Payments', step=0.01, format="%.6f")
    minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
    prc_full_payment = st.number_input(label='Percentage of Full Payment', step=0.01, format="%.6f")
    tenure = st.number_input(label='Tenure', step=1)

    submitted = st.form_submit_button("Submit")

# Main output section
if submitted:
    data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases,
             cash_advance, purchases_frequency, oneoff_purchases_frequency,
             purchases_installment_frequency, cash_advance_frequency, cash_advance_trx,
             purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]
    
    clust = loaded_model.predict(data)[0]
    st.write(f'Prediction: Data belongs to Cluster {clust}')

    # Visualization
    cluster_df1 = df[df['Cluster'] == clust]
    st.subheader(f'Distribution of Features for Cluster {clust}')
    for c in cluster_df1.drop(['Cluster'], axis=1).columns:
        fig, ax = plt.subplots()
        sns.histplot(cluster_df1[c], ax=ax)
        ax.set_title(f'Distribution of {c} for Cluster {clust}')
        st.pyplot(fig)

st.markdown(
    """
    <script>
    document.querySelectorAll('a').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    </script>
    """,
    unsafe_allow_html=True
)
