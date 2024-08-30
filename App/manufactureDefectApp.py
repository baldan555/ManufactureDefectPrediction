import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset from 'manufacture.csv'
@st.cache_data
def load_data():
    df = pd.read_csv('manufacture.csv')
    return df

# Initialize and prepare the dataset
df = load_data()
X = df.drop(columns=['DefectStatus'])
y = df['DefectStatus']

# Split the data for model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model with the best parameters
model = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Streamlit application
st.title('Manufacturing Defect Prediction')

# Initialize session state for storing inserted data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Name'] + X.columns.tolist())

# Sidebar for input and action buttons
st.sidebar.header('Data Entry')
default_values = {
    "ProductionVolume": 5647.606037,
    "ProductionCost": 87.335966,
    "SupplierQuality": 5,
    "DeliveryDelay": 0.638524,
    "DefectRate": 67.628690,
    "QualityScore": 8,
    "MaintenanceHours": 4.692476,
    "DowntimePercentage": 3.577616,
    "InventoryTurnover": 0.055331,
    "StockoutRate": 96.887013,
    "WorkerProductivity": 8,
    "SafetyIncidents": 4652.400275,
    "EnergyConsumption": 0.183125,
    "EnergyEfficiency": 8.097496,
    "AdditiveProcessTime": 164.135870,
    "AdditiveMaterialCost": 1
}

# Input fields with default values
name = st.sidebar.text_input('Name')
inputs = {col: st.sidebar.number_input(col, value=default_values.get(col, 0.0)) for col in X.columns}

# Button to insert data into the table
if st.sidebar.button('Insert Data'):
    if name:
        new_data = pd.DataFrame([[name] + [inputs[col] for col in X.columns]], columns=['Name'] + X.columns.tolist())
        st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
    else:
        st.error("Please enter a name.")

# Show input data in table
st.subheader('Input Data')
st.write(st.session_state.data)

# Predict button below the table
if st.session_state.data.shape[0] > 0:
    if st.button('Predict'):
        features = st.session_state.data.drop(columns=['Name'])
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        st.subheader('Prediction Results')

        # Add predictions and probabilities to the DataFrame
        st.session_state.data['Prediction'] = ['High Defects' if pred == 1 else 'Low Defects' for pred in predictions]
        st.session_state.data['Probability High Defects'] = [prob[1] for prob in probabilities]

        # Visualization of prediction results in card format
        def create_card(title, value, color):
            return f"""
            <div style="
                background-color: {color}; 
                color: white; 
                padding: 20px; 
                margin: 10px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
                <h3>{title}</h3>
                <h2>{value}</h2>
            </div>
            """

        # Display cards
        total_predictions = len(predictions)
        high_defects = sum(pred == 1 for pred in predictions)
        low_defects = total_predictions - high_defects

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(create_card('Total Predictions', total_predictions, '#4CAF50'), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_card('High Defects', high_defects, '#F44336'), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_card('Low Defects', low_defects, '#FFC107'), unsafe_allow_html=True)

        # Display results in card format with probabilities
        st.subheader('Individual Predictions')
        for index, row in st.session_state.data.iterrows():
            name = row['Name']
            prediction = row['Prediction']
            probability = row['Probability High Defects']
            st.markdown(create_card(
                f'Prediction for {name}',
                f'{prediction} (Probability: {probability:.2f})',
                '#2196F3'
            ), unsafe_allow_html=True)
