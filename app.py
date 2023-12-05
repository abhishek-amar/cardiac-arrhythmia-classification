from flask import Flask, redirect, render_template, request
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

@app.route('/', methods = ['POST', 'GET'])
def index():
    error = " "
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return render_template('index.html', error = error)
    return render_template('login.html', error = error)

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    features = [float(x) for x in request.form.values()]
    scaled_features = scaler.transform([features])
    cluster = gmm.predict(scaled_features)
    scaled_features = scaled_features[0, :]
    final = np.concatenate((scaled_features, cluster), axis = 0)
    prediction = stackingclf.predict([final])
    if prediction[0] == 1:
        return render_template('index.html', pred = f"Predicted type of cardiac arrhythmia is: {class_names[int(prediction[0]) - 1]}. No further medical attention required.")
    else:
        return render_template('index.html', pred = f"Predicted type of cardiac arrhythmia is: {class_names[int(prediction[0]) - 1]}. Further medical attention required.")

if __name__ == '__main__':
    app.run(debug = True)