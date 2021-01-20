import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("car_data_km_1000.csv", sep=';')




one_cekis = df["CEKIS"].values.reshape(-1,1)

onehot_encoder_cekis = OneHotEncoder(sparse=False)

OHE_Cekıs = onehot_encoder_cekis.fit_transform(one_cekis)

df["Arkadan"] = OHE_Cekıs[:,0]
df["Onden"] = OHE_Cekıs[:,1]
df["4CEKER"] = OHE_Cekıs[:,2]
print(df)

one_Gear = df["Gear"].values.reshape(-1,1)

onehot_encoder_gear = OneHotEncoder(sparse=False)

OHE_Gear = onehot_encoder_gear.fit_transform(one_Gear)

df["Otomatik"] = OHE_Gear[:,0]
df["Yarı"] = OHE_Gear[:,1]
df["Duz"] = OHE_Gear[:,2]
print(OHE_Cekıs)

one_Fuel = df["Fuel"].values.reshape(-1,1)

onehot_encoder_fuel = OneHotEncoder(sparse=False)

OHE_Fuel = onehot_encoder_fuel.fit_transform(one_Fuel)

df["Benzin"] = OHE_Fuel[:,0]
df["Dizel"] = OHE_Fuel[:,1]
df["LPG"] = OHE_Fuel[:,2]
print(df)

import pickle  # Initialize the flask App

output_1 = open('OHE_cekıs.pkl', 'wb')
output_2 = open('OHE_gear.pkl', 'wb')
output_3 = open('Ohe_fuel.pkl', 'wb')
pickle.dump(onehot_encoder_cekis, output_1)
pickle.dump(onehot_encoder_gear, output_2)
pickle.dump(onehot_encoder_fuel, output_3)

print(onehot_encoder_gear)


from sklearn import preprocessing
import pandas as pd

le_serie = preprocessing.LabelEncoder()
le_brand = preprocessing.LabelEncoder()
le_color = preprocessing.LabelEncoder()
df[['Brand']] = df[['Brand']].apply(le_brand.fit_transform)
df[['Serie']] = df[['Serie']].apply(le_serie.fit_transform)
df[['Color']] = df[['Color']].apply(le_color.fit_transform)
# df['city'] = le.fit(df['city'])


import pickle  # Initialize the flask App

output_1 = open('Brand_Encoder.pkl', 'wb')
output_2 = open('Serie_Encoder.pkl', 'wb')
output_3 = open('Color_Encoder.pkl', 'wb')
pickle.dump(le_brand, output_1)
pickle.dump(le_serie, output_2)
pickle.dump(le_color, output_3)

print(le_color.classes_)
pd.set_option('display.max_columns', 500)
print(df)
X = df.loc[:, ['Brand', 'Serie', 'Color', 'Year', 'KM', 'CC', 'HP',
               'Galeriden', 'GARANTI',
               'Onden', 'Otomatik', 'Yarı','Dizel',
               'LPG']]

y = df.loc[:, ['Price']]


from sklearn.model_selection import train_test_split

y_test: object
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

'''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train.values.ravel())
pickle.dump(rf_reg, open('model.pkl','wb'))
print(X_train)
'''
y_pred = rf_reg.predict(X_test)
print("Accuracy on Traing set: ",rf_reg.score(X_train,y_train))
print("Accuracy on Testing set: ",rf_reg.score(X_test,y_test))
'''
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.

# import libraries
from flask import Flask, render_template, request
import pickle  # Initialize the flask App

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# default page of our web-app
@app.route('/')
def home():
    brands_str = "Alfa Aston Audi BMW Bentley Cadillac Chevrolet Citroen DS Dacia Daewoo Daihatsu Dodge Ferrari Fiat Ford Honda Hyundai Kia Lada Lamborghini Lancia Lexus Lincoln Maserati Mazda Mercedes Mitsubishi Nissan Opel Peugeot Pontiac Renault Saab Seat Skoda Smart Suzuki Volkswagen"
    brands = brands_str.split()
 
    series_str = "- 1 100 106 107 126 2 206 206+ 207 208 3 301 305 306 307 308 323 4 405 406 407 430 458 5 500 508 6 607 626 7 80 807 9.Mar 9.May 900 9000 A1 A3 A4 A5 A6 A7 A8 Accent Accord Albea Almera Altea Alto Applause Arteon Ascona Astra Atos Attrage Automobiles Aveo B-Max Baleno Bluebird Bora Brava Bravo C-Elysee C-Max C1 C2 C3 C4 C5 C8 CTS California Camaro Carisma Ceed Cerato Challenger Charade Citigo City Civic Clarus Clio Colt Continental Cordoba Corsa Corvette Coupe Cruze Cuore DeVille Delta EOS Egea Elantra Epica Escort Espace Espero Evanda Excel Exeo F355 Fabia Favorit Felicia Fiesta Firebird Fleetwood Fluence Focus ForFour ForTwo Forman Fusion Galaxy Gallardo Getz Ghibli Golf GranTurismo Grand Huracan Ibiza Idea Insignia Integra Jazz Jetta Ka Kadett Kalina Kalos Lacetti Laguna Lancer Lanos Lantis Latitude Laurel Legend Leon Liana Linea Lodgy Logan M MX Magentis Marea Mark Martin Maruti Materia Matiz Matrix Megane Meriva Micra Modus Mondeo Mulsanne Mustang New Nexia Note Nubira Octavia Omega Optima Palio Panda Passat Picanto Polo Prelude Pride Primera Pro Pulsar Punto Quattroporte R R8 RC RCZ RS RX Rapid Rekord Rezzo Rio Romeo Roomster S S-Cross S-Max S2000 SX4 Safrane Samara Sandero Saxo Scala Scenic Scirocco Scorpio Sephia Seville Sharan Shuma Siena Sierra Signum Sirion Solenza Sonata Space Spark Stilo Sunny SuperB Swift Symbol TT Talisman Taunus Tempra The Thema Tico Tigra Tipo Toledo Touran Town Twingo Uno VW Vectra Vega Vel Venga Vento Xantia Xsara YRV Ypsilon Z ZX Zafira i i10 i20 i30 i40 ix20"
    series = series_str.split()

    colors_str = "- Altın Bej Beyaz Bordo Füme Gri Kahverengi Kırmızı Lacivert Mavi Mor Pembe Sarı Siyah Turkuaz Turuncu Yeşil Şampanya"
    colors = colors_str.split()

    return render_template('index.html',  brands = brands , series = series ,colors = colors   )





# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [x for x in request.form.values()]
    data = {'Brand': int_features[0],
        'Serie': int_features[1],
        'Color': int_features[2],
        'Year': float(int_features[3]),
        'KM': float(int_features[4]),
        'CC': float(int_features[5]),
        'HP': float(int_features[6]),
        'Galeriden': float(int_features[7]),
        'GARANTI': float(int_features[8]),
        'Drive': int_features[9],
        'Gear': int_features[10],
        'Fuel': int_features[11]
        }
    int_features = pd.DataFrame(data, index=[0])
    print(int_features)

    int_features[['Brand']] = int_features[['Brand']].apply(le_brand.transform)
    int_features[['Serie']] = int_features[['Serie']].apply(le_serie.transform)
    int_features[['Color']] = int_features[['Color']].apply(le_color.transform)
    print(int_features)



    one_cekis = int_features["Drive"].values.reshape(-1,1)


    OHE_Cekıs = onehot_encoder_cekis.transform(one_cekis)
    print(OHE_Cekıs[:,0])
    print(OHE_Cekıs[0])
    print(OHE_Cekıs[0][0])
    int_features["Arkadan"] = OHE_Cekıs[0][1]
    int_features["Onden"] = OHE_Cekıs[0][2]
    int_features["4ceker"] = OHE_Cekıs[0][0]


    print(int_features)

    one_Gear = int_features["Gear"].values.reshape(-1,1)


    OHE_Gear = onehot_encoder_gear.transform(one_Gear)
    print(OHE_Gear)
    print(OHE_Gear[:,0])
    print(OHE_Gear[0])
    print(OHE_Gear[0][0])
    int_features["Otomatik"] = OHE_Gear[0][1]
    int_features["Yarı"] = OHE_Gear[0][2]
    int_features["Duz"] = OHE_Gear[0][0]


    one_Fuel = int_features["Fuel"].values.reshape(-1,1)


    OHE_Fuel = onehot_encoder_fuel.transform(one_Fuel)
    print(OHE_Fuel)
    print(OHE_Fuel[:,0])
    print(OHE_Fuel[0])
    print(OHE_Fuel[0][0])
    int_features["Benzin"] = OHE_Fuel[0][0]
    int_features["Dizel"] = OHE_Fuel[0][1]
    int_features["LPG"] = OHE_Fuel[0][2]


    print(int_features)


    # df['city'] = le.fit(df['city'])
    int_features = int_features.loc[:, ['Brand', 'Serie', 'Color', 'Year', 'KM', 'CC', 'HP',
               'Galeriden', 'GARANTI',
               'Onden', 'Otomatik', 'Yarı','Dizel',
               'LPG']]
    print(int_features)
    final_features = np.array(int_features)
    prediction = rf_reg.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('predict.html', prediction_text='Estimated price of your vehicle is : {:,} ₺'.format(output))




if __name__ == "__main__":
    app.run(debug=True)
'''


try:
    import dill as pickle
except ImportError:
    import pickle

pickle.dump(rf_reg, open('model.pkl','wb'))


#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''