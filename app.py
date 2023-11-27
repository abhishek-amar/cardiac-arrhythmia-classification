from flask import Flask, render_template, request
from joblib import load
import numpy as np
import warnings

warnings.filterwarnings('ignore')
scaler = load('sc.joblib')
gmm = load('gmm.joblib')
stackingclf = load('stackingclf.joblib')
class_names = ["Normal", 
               "Ischemic changes (CAD)", 
               "Old Anterior Myocardial Infraction",
               "Old Inferior Myocardial Infraction",
               "Sinus tachycardy", 
               "Sinus bradycardy", 
               "Ventricular Premature Contraction (PVC)",
               "Supraventricular Premature Contraction",
               "Left Boundle branch block",
               "Right boundle branch block",
               "1.Degree AtrioVentricular block",
               "2.Degree AV block",
               "3.Degree AV block",
               "Left Ventricule hypertrophy",
               "Atrial Fibrillation or Flutter",
               "Others"]
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    features = [float(x) for x in request.form.values()]
    scaled_features = scaler.transform([features])
    print(features)
    cluster = gmm.predict(scaled_features)
    scaled_features = scaled_features[0, :]
    print(scaled_features)
    print(type(scaled_features))
    print(cluster)
    print(type(cluster))
    final = np.concatenate((scaled_features, cluster), axis = 0)
    print(final)
    print(type(final))
    prediction = stackingclf.predict([final])
    print(f"Prediction is: {prediction}")
    print(prediction[0])
    return render_template('index.html', pred = f"Predicted type of cardiac arrhythmia is: {class_names[int(prediction[0]) - 1]}")

if __name__ == '__main__':
    app.run(debug = True)