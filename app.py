from flask import Flask, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)


model = joblib.load('rainfall_svm_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()

     
        input_columns = ['day', 'pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
        input_data = pd.DataFrame(data, index=[0])[input_columns]

        
        prediction = model.predict(input_data)

        
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
      
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
