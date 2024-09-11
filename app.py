# from flask import Flask, render_template, request
# # import pickle
# import numpy as np
# import pandas as pd
# import joblib
#
# # Save model
#
# app = Flask(__name__)
# # reg_model = pickle.load(open('MOD.pkl', 'rb'))
# reg_model = joblib.load('model.pkl')
#
#
# @app.route('/')
# def home():
#     return render_template('HOME.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form data
#     season_names = request.form.get('season_names')
#     crop_names = request.form.get('crop_names')
#     area = float(request.form.get('area'))
#     temperature = float(request.form.get('temperature'))
#     wind_speed = float(request.form.get('wind_speed'))
#     precipitation = float(request.form.get('precipitation'))
#     humidity = float(request.form.get('humidity'))
#     soil_type = request.form.get('soil_type')
#     N = float(request.form.get('N'))
#     P = float(request.form.get('P'))
#     K = float(request.form.get('K'))
#
#     # Create a DataFrame to pass to the model
#     data = pd.DataFrame({
#         'season_names': [season_names],
#         'crop_names': [crop_names],
#         'area': [area],
#         'temperature': [temperature],
#         'wind_speed': [wind_speed],
#         'precipitation': [precipitation],
#         'humidity': [humidity],
#         'soil_type': [soil_type],
#         'N': [N],
#         'P': [P],
#         'K': [K]
#     })
#
#     # Make prediction
#     prediction = reg_model.predict(data)[0]
#
#     # Return the result
#     return render_template('result.html', prediction=prediction)
#
#
# if __name__ == '__main__':
#     app.run(port=8000)

# from flask import Flask, render_template, request
# import pandas as pd
# import pickle
# import numpy as np
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Load the trained model (replace 'model.pkl' with the actual model filename)
# model = pickle.load(open('model.pkl', 'rb'))
#
#
# # Define the home route
# @app.route('/')
# def home():
#     return render_template('home.html')
#
#
# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get form data from the user input
#     season_names = request.form['season_names']
#     crop_names = request.form['crop_names']
#     area = float(request.form['area'])
#     temperature = float(request.form['temperature'])
#     wind_speed = float(request.form['wind_speed'])
#     precipitation = float(request.form['precipitation'])
#     humidity = float(request.form['humidity'])
#     soil_type = request.form['soil_type']
#     N = float(request.form['N'])
#     P = float(request.form['P'])
#     K = float(request.form['K'])
#
#
#     season_name_autumn = 0
#     season_name_kharif = 0
#     season_names_Rabi  = 0
#     season_names_Summer = 0
#     season_names_Whole Year = 0
#     season_names_Winter = 0
#
#     if season_names = 'Autumn':
#         season_name_autumn = 1
#     elif season_name = 'Kharif':
#         season_name_kharif= 1
#     elif season_names = 'Rabi':
#         season_names_Summer = 1
#
#
#
#
#     # Convert categorical inputs to numeric (same encoding as in training)
#     season_names_encoded = {'Autumn ': 0, 'Kharif': 1,'Rabi': 2,'Summer': 3,'Whole Year':4,'Winter':5}[season_names]  # Adjust as needed
#     crop_names_encoded ={
#     'Arecanut': 0,
#     'Arhar/Tur': 1,
#     'Banana': 2,
#     'Barley': 3,
#     'Black pepper': 4,
#     'Blackgram': 5,
#     'Bottle Gourd': 6,
#     'Brinjal': 7,
#     'Cabbage': 8,
#     'Cardamom': 9,
#     'Cashewnut': 10,
#     'Coconut ': 11,
#     'Coriander': 12,
#     'Cowpea(Lobia)': 13,
#     'Cucumber': 14,
#     'Dry chillies': 15,
#     'Dry ginger': 16,
#     'Ginger': 17,
#     'Grapes': 18,
#     'Groundnut': 19,
#     'Jowar': 20,
#     'Khesari': 21,
#     'Korra': 22,
#     'Lentil': 23,
#     'Lemon': 24,
#     'Linseed': 25,
#     'Maize': 26,
#     'Mango': 27,
#     'Masoor': 28,
#     'Mesta': 29,
#     'Moth': 30,
#     'Niger seed': 31,
#     'Onion': 32,
#     'Orange': 33,
#     'Papaya': 34,
#     'Paddy': 35,
#     'Peas  (vegetable)': 36,
#     'Peas & beans (Pulses)': 37,
#     'Pineapple': 38,
#     'Potato': 39,
#     'Pome Fruit': 40,
#     'Pome Granet': 41,
#     'Pulses total': 42,
#     'Rapeseed &Mustard': 43,
#     'Rice': 44,
#     'Ragi': 45,
#     'Safflower': 46,
#     'Samai': 47,
#     'Sesamum': 48,
#     'Small millets': 49,
#     'Sannhamp': 50,
#     'Soyabean': 51,
#     'Sugarcane': 52,
#     'Sunflower': 53,
#     'Sweet potato': 54,
#     'Tapioca': 55,
#     'Tomato': 56,
#     'Tobacco': 57,
#     'Total foodgrain': 58,
#     'Turmeric': 59,
#     'Urad': 60,
#     'Varagu': 61,
#     'Wheat': 62,
#     'Other Cereals & Millets': 63,
#     'Other Fresh Fruits': 64,
#     'Other Kharif pulses': 65,
#     'Other Rabi pulses': 66,
#     'Other Vegetables': 67,
#     'other fibres': 68,
#     'other misc. pulses': 69,
#     'other oilseeds': 70,
#     'Jute': 71
# }[crop_names]  # Adjust based on your crops
#     soil_type_encoded = soil_types_encoded = {
#     'chalky': 0,
#     'clay': 1,
#     'loamy': 2,
#     'peaty': 3,
#     'sandy': 4,
#     'silt': 5,
#     'silty': 6
# }[soil_type]  # Adjust based on soil types
#
#     # Create a DataFrame for the input features
#     input_data = pd.DataFrame([[season_names_encoded, crop_names_encoded, area, temperature, wind_speed,
#                                 precipitation, humidity, soil_type_encoded, N, P, K]],
#                                columns=['season_names_Autumn','season_names_Kharif','season_names_Rabi','season_names_Summer','season_names_Whole Year','season_names_Winter', 'crops_name_encoded', 'area', 'temperature', 'wind_speed',
#                                         'precipitation', 'humidity', 'soil_type_chalky', 'soil_type_clay', 'soil_type_loamy', 'soil_type_peaty', 'soil_type_sandy' 'soil_type_silt' 'soil_type_silty', 'N', 'P', 'K'])
#
#     # Predict the production using the model
#     prediction = model.predict(input_data)[0]
#
#     # Render the result on a new page
#     return render_template('result.html', prediction=prediction)
#
#
# if __name__ == "__main__":
#     app.run(port=8000)


from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (replace 'model.pkl' with the actual model filename)
model = pickle.load(open('model.pkl', 'rb'))


# Define the home route
@app.route('/')
def home():
    return render_template('home.html')


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the user input
    season_names = request.form['season_names']
    crop_names = request.form['crop_names']
    area = float(request.form['area'])
    temperature = float(request.form['temperature'])
    wind_speed = float(request.form['wind_speed'])
    precipitation = float(request.form['precipitation'])
    humidity = float(request.form['humidity'])
    soil_type = request.form['soil_type']
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])

    # One-hot encoding for seasons
    season_name_autumn = 0
    season_name_kharif = 0
    season_name_rabi = 0
    season_name_summer = 0
    season_name_whole_year = 0
    season_name_winter = 0

    if season_names == 'Autumn':
        season_name_autumn = 1
    elif season_names == 'Kharif':
        season_name_kharif = 1
    elif season_names == 'Rabi':
        season_name_rabi = 1
    elif season_names == 'Summer':
        season_name_summer = 1
    elif season_names == 'Whole Year':
        season_name_whole_year = 1
    elif season_names == 'Winter':
        season_name_winter = 1

    # One-hot encoding for soil types
    soil_type_chalky = 0
    soil_type_clay = 0
    soil_type_loamy = 0
    soil_type_peaty = 0
    soil_type_sandy = 0
    soil_type_silt = 0
    soil_type_silty = 0

    if soil_type == 'chalky':
        soil_type_chalky = 1
    elif soil_type == 'clay':
        soil_type_clay = 1
    elif soil_type == 'loamy':
        soil_type_loamy = 1
    elif soil_type == 'peaty':
        soil_type_peaty = 1
    elif soil_type == 'sandy':
        soil_type_sandy = 1
    elif soil_type == 'silt':
        soil_type_silt = 1
    elif soil_type == 'silty':
        soil_type_silty = 1

    label_dict = {'Arecanut': 0,   'Other Kharif pulses': 49,    'Rice': 62,'Banana': 3,    'Cashewnut': 13,'Coconut ': 16,
                  'Dry ginger': 22, 'Sugarcane': 70,'Sweet potato': 72,'Tapioca': 73,
                  'Black pepper': 7,'Dry chillies': 21,'other oilseeds': 83,'Turmeric': 77,
                  'Maize': 36,'Moong(Green Gram)': 40,'Urad': 78,'Arhar/Tur': 1,
                  'Groundnut': 27,'Sunflower': 71,'Bajra': 2,'Castor seed': 14,
                  'Cotton(lint)': 18,'Horse-gram': 28,'Jowar': 29,'Korra': 32,'Ragi': 60,'Tobacco': 74,'Gram': 25,'Wheat': 80,
                  'Masoor': 38,'Sesamum': 67,'Linseed': 35,'Safflower': 63,
                  'Onion': 44,'other misc. pulses': 82,'Samai': 64,'Small millets': 68,
                  'Coriander': 17,'Potato': 58,'Other  Rabi pulses': 46,'Soyabean': 69,
                  'Beans & Mutter(Vegetable)': 5,'Bhindi': 6,'Brinjal': 10,'Citrus Fruit': 15,'Cucumber': 20,'Grapes': 26,'Mango': 37,'Orange': 45,'other fibres': 81,
                  'Other Fresh Fruits': 48,'Other Vegetables': 50,'Papaya': 52,'Pome Fruit': 56,'Tomato': 75,
                  'Mesta': 39,'Cowpea(Lobia)': 19,'Lemon': 33,'Pome Granet': 57,'Sapota': 66,'Cabbage': 11,
                  'Rapeseed &Mustard': 61,'Peas  (vegetable)': 53,'Niger seed': 42,
                  'Bottle Gourd': 9,'Varagu': 79,'Garlic': 23,'Ginger': 24,'Oilseeds total': 43,'Pulses total': 59,'Jute': 30,
                  'Peas & beans (Pulses)': 54,'Blackgram': 8,'Paddy': 51,'Pineapple': 55,'Other Cereals & Millets': 47,'Barley': 4,'Total foodgrain': 76,
                  'Lentil': 34,'Sannhamp': 65,'Khesari': 31,'Cardamom': 12,'Moth': 41}

    for key, value in label_dict.items():
        if key == crop_names:
            crop_code = value

    # Create a DataFrame for the input features
    input_data = [[season_name_autumn, season_name_kharif, season_name_rabi, season_name_summer,
                                        season_name_whole_year, season_name_winter, crop_code, area,
                                        temperature,wind_speed,precipitation,humidity,soil_type_chalky,
                                        soil_type_clay,soil_type_loamy,soil_type_peaty,soil_type_sandy,
                                        soil_type_silt,soil_type_silty,N,P,K]]

    # Predict the production using the model
    prediction = model.predict(input_data)[0]

    # Render the result on a new page
    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(port=8000)
