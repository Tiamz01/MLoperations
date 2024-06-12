# import predict

import requests

ride = {
    'PULocationID': 10,
    'DOLocationID': 15,
    'duration': 10
}

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=ride)
print(response.json())


# features = predict.prepare_features(ride)
# pred = predict.predict(ride)
# print(pred)