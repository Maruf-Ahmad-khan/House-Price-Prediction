import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('linearregression.pkl','rb')) 
# run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    
    r1 = float(request.args.get('SqFt'))
    r2 = float(request.args.get('bedrooms'))
    r3 = float(request.args.get('Bathrooms'))
    r4 = float(request.args.get('Offers'))
    r5 = float(request.args.get('bricks'))
    r6 = float(request.args.get('Neighborhood'))
    
    result=np.array([r1,r2,r3,r4,r5,r6]).reshape(1,-1)
    prediction = model.predict(result)
    
    
        
    return render_template('index.html', prediction_text='Regression Model has predicted Price for given requirements is : {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)