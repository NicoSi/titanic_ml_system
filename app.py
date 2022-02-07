import sklearn
import joblib
import pandas as pd 
import numpy as np
from flask import request, Flask


pipeline = joblib.load('model_final.pkl')

#Démarrage de l'appli Flask
app = Flask('__name__')

@app.route('/')
def index():
  return "<h1>Bienvenue dans notre API. Utilisez /predict en POST pour faire des prédictions <h1>"


#Route pour tester l'api (ping)
@app.route('/ping', methods=['GET'])
def ping():
  return('pong', 200)

#Route pour réaliser des prédictions 
@app.route('/predict', methods=['POST'])
def predict():

  df = pd.DataFrame(request.json)
  result = pipeline.predict(df)[0]

  return(str(result),201)

if __name__ == "__main__":
  app.run(host='0.0.0.0')
