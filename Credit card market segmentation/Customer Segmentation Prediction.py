import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

# Function to load model and data
def load_data_and_model():
    filename = 'final_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    df = pd.read_csv("clustered_data.csv")
    return loaded_model, df

# Function to create Streamlit form with styled elements
def create_input_form():
    st.markdown(
        """
        <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #121212; /* Dark background color */
            color: #e0e0e0; /* Light text color */
            margin: 0; /* Remove default margin */
            padding: 0; /* Remove default padding */
        }
        .stTextInput, .stNumberInput {
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
            border: 1px solid #333; /* Darker border */
            background-color: #333; /* Darker background for inputs */
            color: #e0e0e0; /* Light text color */
        }
        .stButton {
            background-color: #4CAF50; /* Green button background */
            color: white; /* White button text */
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            display: block;
            margin: auto;
        }
        .stButton:hover {
            background-color: #45a049; /* Darken button on hover */
        }
        .stPlotlyChart {
            margin: auto; /* Center align plotly charts */
        }
        .stApp {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
            background-color: #1f1f1f; /* Slightly lighter background for content */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            text-align: center; /* Center align content */
        }
        .plotly .modebar-container {
            display: none !important; /* Hide mode bar */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Customer Segmentation Prediction")  # Change the title here
    st.markdown("---")  # Horizontal rule for separation

    with st.form("my_form"):
        fields = [
            ('Balance', 'number', 0.001, "%.6f"),
            ('Balance Frequency', 'number', 0.001, "%.6f"),
            ('Purchases', 'number', 0.01, "%.2f"),
            ('OneOff Purchases', 'number', 0.01, "%.2f"),
            ('Installments Purchases', 'number', 0.01, "%.2f"),
            ('Cash Advance', 'number', 0.01, "%.6f"),
            ('Purchases Frequency', 'number', 0.01, "%.6f"),
            ('OneOff Purchases Frequency', 'number', 0.1, "%.6f"),
            ('Purchases Installments Frequency', 'number', 0.1, "%.6f"),
            ('Cash Advance Frequency', 'number', 0.1, "%.6f"),
            ('Cash Advance Trx', 'number', 1, None),
            ('Purchases TRX', 'number', 1, None),
            ('Credit Limit', 'number', 0.1, "%.1f"),
            ('Payments', 'number', 0.01, "%.6f"),
            ('Minimum Payments', 'number', 0.01, "%.6f"),
            ('PRC Full Payment', 'number', 0.01, "%.6f"),
            ('Tenure', 'number', 1, None)
        ]
        
        data = []
        for label, input_type, step, format_str in fields:
            if format_str:
                value = st.number_input(label=label, step=step, format=format_str, key=label.lower().replace(' ', '_'))
            else:
                value = st.number_input(label=label, step=step, key=label.lower().replace(' ', '_'))
            data.append(value)
        
        submitted = st.form_submit_button("Submit")
    
    if submitted:
        clust = loaded_model.predict([data])[0]
        st.markdown("---")  # Horizontal rule for separation
        st.header(f'Data Belongs to Cluster {clust}')
        
        cluster_df = df[df['Cluster'] == clust].drop(columns=['Cluster'], axis=1)
        
        for column in cluster_df.columns:
            fig = px.histogram(cluster_df, x=column, title=f'Distribution of {column}')
            fig.update_layout(title={'text': f'Distribution of {column}', 'x': 0.43})  # Center align title
            st.plotly_chart(fig, use_container_width=True)  # Ensure plots use full width

# Main execution flow
if __name__ == '__main__':
    loaded_model, df = load_data_and_model()
    st.set_page_config(page_title="Customer Segmentation Prediction", layout="wide")
    
    create_input_form()
