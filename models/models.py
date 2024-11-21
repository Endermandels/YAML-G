from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import pickle # TODO: Pickle final model


def get_data() -> pd.DataFrame:
    return None

def create_pipe(data: pd.DataFrame) -> Pipeline:
    return None

def train_model(data: pd.DataFrame, pipe: Pipeline):
    pass

def main():
    data = get_data()
    pipe = create_pipe(data)
    train_model(data, pipe)

if __name__ == '__main__':
    main()