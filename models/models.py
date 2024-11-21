from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import pickle # TODO: Pickle final model
import ast
import re

TARGET = 'is_legendary'

def preprocess_data(path: str) -> pd.DataFrame:
    """
    ### Params
        path: Path to data set
    
    ### Returns
        DataFrame containing both X and Y features
    """
    data = pd.read_csv(path)
    data = data.drop(columns=['classfication', 'japanese_name', 'name', 'generation'])
    
    # NOTE: Some abilities are duplicated
    # NOTE: Single abilities are not surrounded in "" while multiple ones are
    
    # TODO: Change capture_rate to numerical (change 30 (Meteorite)255 (Core) to larger number)
    # Make it generalizable
    
    categorical_columns = data.select_dtypes(include=['object']).columns.drop(labels=['abilities'])
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    ability_column = 'abilities'
    
    data[categorical_columns] = data[categorical_columns].fillna('Missing')
    data[ability_column] = data[ability_column].fillna('[]')
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    # TODO: Create separate column for each ability (manually)
   
    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical data and add the encoded data
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Get the correct feature names from the encoder
    categorical_columns = encoder.get_feature_names_out(categorical_columns)
    
    print(data)

    return data

def create_pipe(data: pd.DataFrame) -> Pipeline:
    return None

def train_model(data: pd.DataFrame, pipe: Pipeline):
    pass

def main():
    data = preprocess_data('../pokemon/updated_data.csv')
    pipe = create_pipe(data)
    train_model(data, pipe)

if __name__ == '__main__':
    main()