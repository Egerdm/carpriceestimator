#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from jinja2 import Template
import pandas as pd

#Initialize the flask App
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

@app.route('/about')
def about():

    return render_template('about.html',    )

  

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
