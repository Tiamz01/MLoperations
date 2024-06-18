#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import uuid



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def read_data(filename):
    taxi_type = sys.argv[1] #'yellow'
    year = int(sys.argv[2]) #2023
    month = int(sys.argv[3]) #3

    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def prepare_df_dicts(df:pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def apply_model(input_file, model, output_file):

    df = read_data(input_file)
    dicts = prepare_df_dicts(df)
    
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Coverting result to dataframe
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    print(f"The mean of predicted duration:{df_result['predicted_duration'].mean()}")

    # saving the dataframe as parquet file
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
   


def run():

    taxi_type = sys.argv[1] #'yellow'
    year = int(sys.argv[2]) #2023
    month = int(sys.argv[3]) #3

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'
    
    apply_model(
        input_file=input_file,
        model=model,
        output_file=output_file
    )




if __name__== "__main__":
    run()




