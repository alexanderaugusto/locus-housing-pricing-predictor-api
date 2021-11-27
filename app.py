from flask import Flask, request, jsonify
from modules.ProcessDataAndPredict import ProcessDataAndPredict
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
   
    try:
        predictor = ProcessDataAndPredict(data['data'])
        predictor.process()
        prediction = predictor.predict()
        return { 'prediction': prediction }, 200
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500

if __name__ == '__main__':
    environment = os.environ.get('ENV', 'development')

    if(environment == 'development'):
        print('Running in development mode')
        app.config.from_object('config.DevelopmentConfig')
    else:
        print('Running in production mode')
        app.config.from_object('config.ProductionConfig')

    debug = app.config.get('DEBUG')
    port = app.config.get('PORT')

    app.run(debug=debug, port=port)
    