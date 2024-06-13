#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip freeze | grep scikit-learn')


get_ipython().system('python -V')



import pickle
import pandas as pd
import uuid



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)



categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


year = 2023
month = 3
taxi_type = 'yellow'

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


df = read_data(input_file)
df

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)
y_pred.std()


year = 2023
month = 3

# Generating ride ID as string
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')



# Coverting result to dataframe
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# saving the dataframe as parquet file
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

df_result




