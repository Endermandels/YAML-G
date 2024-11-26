import sys
import pickle
import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator

# Classes needed by the pickled object

class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        return xs[self.columns]

class TransformData(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, xs, ys, **params):
        return self
    
    def transform(self, xs):
        result = xs.apply(self.func)
        return result
    
# Functions needed by the pickled object
    
def get_identity(x):
    return x

def get_sqrt(x):
    return np.where(x > 0, np.sqrt(x), 0)

# Deserialize and return serialized objects

def unpickle(model_file : str):
    try:
        with open(model_file, 'rb') as pkl_file:
            model = pickle.load(pkl_file)
            return model
    except Exception as e:
        print(f"Error unpickling: {e}")
        sys.exit(1)

# Obtain new data to test the model with.
# The new data will be modified in the same way that the original data was for training.
# TODO eventually this data will be taken in from a GUI.

def get_test_data(xs):

    base_egg_steps = random.randint(10000, 20000)

    test_data = {
        'abilities' : [['Pressure', 'Pickpocket']],
        'base_egg_steps' : [base_egg_steps],
        'capture_rate' : [50],
        'hp' : [50],
        'speed' : [100],
    }

    # create a dataframe and get it in the expected format
    test_df = pd.DataFrame(test_data)
    mlb = MultiLabelBinarizer()
    abilities_data = pd.DataFrame(  mlb.fit_transform(test_df['abilities']), columns=mlb.classes_, index=test_df.index)
    test_df = test_df.drop(columns=['abilities'], axis=1)
    test_df = pd.concat([test_df, abilities_data], axis=1)
    test_df = test_df.reindex(columns=xs.columns, fill_value=0)
    return test_df

# Given a pandas dataframe in the a format used for training, predict if the pokemon is legendary

def predict_is_legendary(df, model):
    result = model.predict(df)
    if(result == [0]):
        print("Predicted not legendary!")
    else:
        print("Predicted legendary!")

def main():
    model = unpickle("model.pickle")
    xs = unpickle("xs.pickle")

    for _ in range(0, 5):
        print("Testing another pokemon...")
        test_df = get_test_data(xs)
        predict_is_legendary(test_df, model)
        time.sleep(2)

    print("Program terminating")

if __name__ == '__main__':
    main()
