import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import sys

app = Flask(__name__)

# Load model and scaler with error handling
try:
    with open('housepred.pkl','rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")
    print("\nPlease ensure:")
    print("1. housepred.pkl and scaler.pkl files exist in the current directory")
    print("2. You have compatible scikit-learn version installed (try: pip install scikit-learn==1.0.2)")
    print("3. If the error persists, retrain the model using the Jupyter notebook")
    sys.exit(1)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        a_data = (np.array(list(data.values())).reshape(1,-1))
        new_data = scaler.transform(a_data)
        output = model.predict(new_data)
        print(output[0])
        return jsonify({
            'prediction': float(output[0]),
            'formatted_price': format_price(output[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def format_price(price):
    """Format price in a user-friendly way"""
    price_value = float(price)
    if price_value >= 1000:
        return f"${price_value:,.0f}K (${price_value*1000:,.0f})"
    return f"${price_value:,.2f}K"

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=[float(x) for x in request.form.values()]
        final_input=scaler.transform(np.array(data).reshape(1,-1))
        print(final_input)
        output=model.predict(final_input)[0]
        formatted_price = format_price(output)
        return render_template("home.html",prediction_text=f"🏠 Predicted House Price: {formatted_price}")
    except ValueError as e:
        return render_template("home.html",prediction_text="❌ Error: Please enter valid numbers for all fields")
    except Exception as e:
        return render_template("home.html",prediction_text=f"❌ Error: {str(e)}")


if __name__ =="__main__":
    app.run(debug=True)