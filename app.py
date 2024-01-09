from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
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

'''
To create database, run the following if database does not exist:
In the terminal:
sqlite3 database.db
CREATE TABLE User (
username TEXT PRIMARY KEY,
password TEXT NOT NULL
);

Or if the datavase already exists:
In the terminal:
sqlite3
.open database.db
CREATE TABLE User (
username TEXT PRIMARY KEY,
password TEXT NOT NULL
);
'''

app = Flask(__name__)
db_path = os.path.join(os.path.dirname(__file__), 'database.db')
db_uri = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
db = SQLAlchemy(app)

class User(db.Model):
    username = db.Column(db.String(20), primary_key = True)
    password = db.Column(db.String(20))

@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        user = User.query.filter_by(username = request.form['username']).first()
        if user is None:
            return render_template('login.html', error = "Invalid Credentials. Please try again.")
        else:
            if check_password_hash(user.password, request.form['password']):
                return render_template('index.html')
            else:
                return render_template('login.html', error = "Invalid Credentials. Please try again.")
    return render_template('login.html')

@app.route('/signup', methods = ['POST', 'GET'])
def signup():
    if request.method == 'POST':
        user = User.query.filter_by(username = request.form['username']).first()
        if user is not None:
            return render_template('signup.html', message = "Username already exists.")
        hashed_password = generate_password_hash(request.form['password'], method='sha256')
        new_user = User(username = request.form['username'], password = hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return render_template('signup.html', message = "Sign up successful.")
    return render_template('signup.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    features = [float(x) for x in request.form.values()]
    if len(features) != 10:
        return render_template('login.html')
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