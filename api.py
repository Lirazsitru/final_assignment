from flask import Flask, render_template, request
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle

app = Flask(__name__)



preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    model = pickle.load(open('trained_model.pkl', 'rb'))
    # Retrieve the user inputs
    city = request.form.get('City')
    property_type = request.form.get('type')
    room_number = request.form.get('room_number')
    area = request.form.get('Area')
    room_number = float(room_number)
    area = float(area)
    has_elevator = request.form.get('hasElevator ')
    has_parking = request.form.get('hasParking ')
    has_storage = request.form.get('hasStorage ')
    condition = request.form.get('condition ')
    has_air_condition = request.form.get('hasAirCondition ')
    has_balcony = request.form.get('hasBalcony ')
    has_mamad = request.form.get('hasMamad ')
    handicap_friendly = request.form.get('handicapFriendly ')

    # Create a DataFrame from the user inputs
    user_data = pd.DataFrame({
        'City': [city],
        'type': [property_type],
        'room_number': [room_number],
        'Area': [area],
        'hasElevator ': [has_elevator],
        'hasParking ': [has_parking],
        'hasStorage ': [has_storage],
        'condition ': [condition],
        'hasAirCondition ': [has_air_condition],
        'hasBalcony ': [has_balcony],
        'hasMamad ': [has_mamad],
        'handicapFriendly ': [handicap_friendly]
    })



    # Make the prediction using the model
    predicted_price = model.predict(user_data)[0]

    return f"The predicted price for this listing is: {predicted_price:.2f}"




if __name__ == '__main__':
    app.run(debug=True)
