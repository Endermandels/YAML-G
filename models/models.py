from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import pickle

TARGET = 'is_legendary'

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
    
def get_identity(x):
    return x

def get_sqrt(x):
    return np.where(x > 0, np.sqrt(x), 0)

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
    data = data.drop(columns=['classfication', 'japanese_name', 'name', 'generation',
                              'against_bug', 'against_dark', 'against_dragon', 'against_electric',
                              'against_fairy', 'against_fight', 'against_fire', 'against_flying',
                              'against_ghost', 'against_grass', 'against_ground', 'against_ice',
                              'against_normal', 'against_poison', 'against_psychic', 'against_rock',
                              'against_steel', 'against_water', 'type1',  'attack', 'defense',
                               'sp_attack', 'sp_defense', 'type2', 'pokedex_number', 'percentage_male',
                              'abilities', 'speed', 'hp', 'experience_growth', 'base_happiness', 'height_m', 'weight_kg'  ])
    

    print(data)
    
    # Make capture_rate a numeric data type column
    data['capture_rate'] = data['capture_rate'].str.extract('(\\d+)').astype(np.int64)
    
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

    return data

def create_pipe() -> Pipeline:
    """
    Create Pipeline with the following steps:
        column_select
        PCA
        transform_data
        regression
    
    ### Returns
        Pipline
    """
    steps = [
        ('column_select', SelectColumns(['capture_rate']))
        # , ('PCA', PCA(n_components=2))
        , ('transform_data', TransformData(get_identity))
        , ('regression', GradientBoostingClassifier(max_depth=2))
    ]
    return Pipeline(steps)

def create_grid_search(data: pd.DataFrame, pipe: Pipeline) -> GridSearchCV:
    """
    Create GridSearchCV.

    ### Params
        data: DataFrame containing X and Y features
        pipe: Pipeline
    
    ### Returns
        GridSearchCV
    """
    grid = {
        'column_select__columns': [
            list(data.drop(columns=TARGET).columns)
        ],
        # 'PCA__n_components': [
        #     1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        # ],
        'transform_data__func': [
            #lambda x: np.where(x > 0, np.sqrt(x), 0)
            get_sqrt
            #, lambda x: x
            , get_identity
            
            , np.square
        ],
        'regression__max_depth': [
            2,3,4,5,6,7,8,9
        ],
        'regression__learning_rate': [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]
    }
    return GridSearchCV(pipe, grid, scoring='f1', n_jobs=-1, cv=5)

def train_model(data: pd.DataFrame, search: GridSearchCV):
    """
    Train model in GridSearchCV.
    
    ### Params
        data:   DataFrame containing X and Y features
        search: GridSearchCV
    
    ### Returns
        Model with best hyperparameters
    """
    xs = data.drop(columns=[TARGET])
    ys = data[TARGET]

    search.fit(xs, ys)

    print('Gradient boosting:')
    print('F1-Score:', search.best_score_)
    print('Best params:', search.best_params_)
    return search.best_estimator_

def save_model(save_fn: str, model):
    """
    Save model to save_fn.
    """
    try:
        with open(save_fn, 'wb') as file:
            pickle.dump(model, file)
            print("Pickle saved to file: ", save_fn)
    except Exception as e:
        print(f"Error while saving model: {e}")

def main():
    data = preprocess_data('../pokemon/updated_data.csv')
    pipe = create_pipe()
    search = create_grid_search(data, pipe)
    model = train_model(data, search)
    save_model('model.pickle', model)
    save_model('xs.pickle', data.drop(columns=[TARGET]) )

if __name__ == '__main__':
    main()
    