from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import pickle # TODO: Pickle final model

TARGET = 'is_legendary'

def preprocess_data(path: str) -> pd.DataFrame:
    """
    Drops irrelevant data.
    Converts capture_rate to numerical data.
    One Hot Encodes categorical data.
    
    ### Params
        path: Path to data set
    
    ### Returns
        DataFrame containing both X and Y features
    """
    data = pd.read_csv(path)
    data = data.drop(columns=['classfication', 'japanese_name', 'name', 'generation'])
    
    # Make capture_rate a numeric data type column
    data['capture_rate'] = data['capture_rate'].str.extract('(\d+)').astype(np.int64)
    
    ability_column = 'abilities'
    categorical_columns = \
        data.select_dtypes(include=['object']).columns.drop(labels=[ability_column])
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    data[categorical_columns] = data[categorical_columns].fillna('Missing')
    data[ability_column] = data[ability_column].fillna('[]')
    data[numeric_columns] = data[numeric_columns].fillna(0)
    
    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(  encoded_columns
                              , columns=encoder.get_feature_names_out(categorical_columns))

    # Drop the original categorical data and add the encoded data
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Get the correct feature names from the encoder
    categorical_columns = encoder.get_feature_names_out(categorical_columns)
    
    # Convert stringified lists to actual lists
    data['abilities'] = data['abilities'].apply(lambda x: eval(x))

    # Convert abilities to multiple one-hot-encoded columns
    mlb = MultiLabelBinarizer()
    abilities_data = pd.DataFrame(  mlb.fit_transform(data['abilities'])
                                  , columns=mlb.classes_
                                  , index=data.index)
    
    # Drop abilities column and add encoded data
    data = data.drop(ability_column, axis=1)
    data = pd.concat([data, abilities_data], axis=1)

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