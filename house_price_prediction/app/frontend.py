import streamlit as st
import joblib
import pandas as pd

def create_features(X):
    X = X.copy()

    X['total_rooms'] = X['bedrooms'] + X['bathrooms']
    X['room_size'] = X['size_sqft'] / (X['total_rooms'] + 1)
    X['luxury_score'] = (X['has_pool'] + X['has_garden']) * X['size_sqft'] / 1000

    X['is_new'] = (X['age_years'] < 5).astype(int)
    X['is_old'] = (X['age_years'] > 30).astype(int)

    X['is_small'] = (X['size_sqft'] < 1500).astype(int)
    X['is_large'] = (X['size_sqft'] > 3000).astype(int)

    return X


model = joblib.load('house_price_model_v1.pkl')
pipeline = model['pipeline']

st.set_page_config(page_title="House Price Predictor", layout='centered')

st.title('House Price Prediction App')
st.write('Enter the house details')

feature1 = st.slider('House Size (in sqft)', 500, 10000)
feature2 = st.slider('Bedrooms', 1, 7)
feature3 = st.slider('Bathrooms', 1, 4)
feature4 = st.number_input('Age of the House (in years)', min_value = 0, max_value = None)

feature5 = st.radio('Condition', ['Poor', 'Fair', 'Good', 'Excellent'])
feature6 = st.radio('Location', ['urban', 'suburban', 'rural'])
feature7 = st.radio('Garage', ['Yes', 'No'])
feature8 = st.radio('Garden', ['Yes', 'No'])
feature9 = st.radio('Pool', ['Yes', 'No'])


condition_map = {'Poor' : 0, 'Fair' : 1, 'Good' : 2, 'Excellent' : 3}
location_map = {'urban' : 0, 'suburban' : 1, 'rural' : 2}
yes_no_map = {'Yes' : 1, 'No': 0}

feature5 = condition_map[feature5]
feature6 = location_map[feature6]
feature7 = yes_no_map[feature7]
feature8 = yes_no_map[feature8]
feature9 = yes_no_map[feature9]

input_df = pd.DataFrame([{'size_sqft' : feature1, 'bedrooms' : feature2, 'bathrooms' : feature3, 'age_years' : feature4, 'garage' : feature5, 'location' : feature6, 'condition' : feature7, 'has_pool' : feature8, 'has_garden' : feature9}])

if st.button('Predict Price'):
    prediction = pipeline.predict(input_df)
    st.success(f'Predicted Price: {prediction[0]:,.2f}')
