import predict

ride = {
    'PULocationID': 10,
    'DOLocationID': 15,
    'duration': 10
}


features = predict.prepare_features(ride)
pred = predict.predict(ride)
print(pred)