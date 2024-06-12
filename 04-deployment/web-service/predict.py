import pickle
from flask import Flask, request, jsonify

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features['PULocationID'] = ride['PULocationID']
    features['DOLocationID'] = ride['DOLocationID']
    features['duration'] = ride['duration']

    return features

# Use the dictvectorizer and model from the pickle file to turn features to matrix and use the model to make predictions
def predict(features):
    x= dv.transform(features)
    preds = model.predict(x)
    return preds[0]


app = Flask('distance_prediction')


#This is used to create an endpoint that will be connected to flask to make the app accessible through http request
@app.route('/predict', methods=['POST'])
def predict_endpiont():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'trip_distance': pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
