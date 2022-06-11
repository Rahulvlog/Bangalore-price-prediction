from flask import Flask, render_template, request
from matplotlib.backend_bases import LocationEvent
import pandas as pd
import pickle



app = Flask(__name__)
housing = pd.read_csv('CLeaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    location = sorted(housing['location'].unique())
    return render_template('index.html', locations = location)
    
@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = request.form.get('BHK')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(locations, bhk, bath, sqft)
    input = pd.DataFrame([[locations, sqft, bath, bhk]],columns=['location', 'total_sqft', 'bath', 'BHK'])
    prediction = pipe.predict(input)[0]


     
    return str(prediction)
if __name__ == "__main__":
    app.run(debug=True, port=5001)