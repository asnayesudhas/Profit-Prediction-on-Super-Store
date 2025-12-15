import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv(r"C:\Users\madhu\Downloads\superstore.csv")

# Select required columns
data = data[['Category', 'Sub-Category', 'Sales', 'Quantity', 'Profit']]

# -------------------------------
# Encode categorical columns
# -------------------------------
le_category = LabelEncoder()
le_subcategory = LabelEncoder()

data['Category'] = le_category.fit_transform(data['Category'])
data['Sub-Category'] = le_subcategory.fit_transform(data['Sub-Category'])

# -------------------------------
# Split data
# -------------------------------
X = data[['Category', 'Sub-Category', 'Sales', 'Quantity']]
y = data['Profit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Train Gradient Boosting Model
# -------------------------------
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 Superstore Profit Prediction")
st.write("Enter product details to predict **Profit**")

# Inputs
category = st.text_input("Category")
sub_category = st.text_input("Sub-Category")
sales = st.number_input("Sales", min_value=0.0, step=1.0)
quantity = st.number_input("Quantity", min_value=1, step=1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Profit"):

    if category not in le_category.classes_:
        st.error("Category not found in dataset")
        st.stop()

    if sub_category not in le_subcategory.classes_:
        st.error("Sub-Category not found in dataset")
        st.stop()

    category_encoded = le_category.transform([category])[0]
    subcategory_encoded = le_subcategory.transform([sub_category])[0]

    input_data = [[category_encoded, subcategory_encoded, sales, quantity]]
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Profit: {prediction[0]:.2f}")
