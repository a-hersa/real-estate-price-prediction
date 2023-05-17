import os
import sys
import pickle

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

# define the transformer
class RealEstatePreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        print('Initialising transformer...')
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        print('Transforming..')
        X=X.drop(labels=['time', 'tag'], axis=1)

        # type
        X.rename(columns={'type': 'deal_type'}, inplace=True)

        # sqrm
        X['sqrm'] = X['sqrm'].replace('[^0-9]', np.nan, regex=True)
        X = X.dropna(subset=['sqrm'])
        X['sqrm'] = X['sqrm'].astype(int)

        # property_type_encoded & dropping duplicates
        X['property_type'] = [x.split()[0] for x in X['title']]
        X = X[X['property_type'] != 'Estudio']
        X = X[X['property_type'] != 'Finca']
        X=X.drop(labels=['title'], axis=1)
        X=X.drop_duplicates()
        # ptype_price_sqrm = X.groupby('property_type', as_index=False).apply(lambda x: pd.Series({'property_type_encoded':x['price'].sum() / x['sqrm'].sum()})).set_index('property_type')['property_type_encoded'].to_dict()
        # X['property_type_encoded'] = X['property_type'].map(ptype_price_sqrm)

        # floor
        X['floor'] = X['floor'].replace('-', '-1')
        X['floor'] = X['floor'].astype(float)
        X.loc[X['property_type'].isin(['Casa', 'Castillo', 'Chalet', 'Cortijo', 'Finca', 'Masía', 'Torre']), 'floor'] = '0'
        X['floor'] = X['floor'].astype(float)
        X['floor'] = X['floor'].fillna(X.groupby('property_type')['floor'].transform('mean'))
        X['floor'] = X['floor'].astype(int)

        # elevator
        X.loc[X['property_type'].isin(['Casa', 'Castillo', 'Chalet', 'Cortijo', 'Finca', 'Masía', 'Torre']), 'elevator'] = '0'
        X['elevator'] = X['elevator'].astype(float)
        X['elevator'] = X['elevator'].fillna(X.groupby('property_type')['elevator'].transform('mean'))
        X['elevator'] = X['elevator'].astype(int)

        # surface
        X['surface'] = X['surface'].fillna('0')
        X['surface'] = X['surface'].replace('outdoor', '1')
        X['surface'] = X['surface'].astype(int)

        # rooms
        X = X.dropna(subset=['rooms'])
        X['rooms'] = X['rooms'].astype(int)

        # location_encoded
        X['province'] = X['province'].fillna('empty')
        X['county'] = X['county'].fillna('empty')
        X['city'] = X['city'].fillna('empty')
        X['area'] = X['area'].fillna('empty')
        X['neighborhood'] = X['neighborhood'].fillna('empty')
        X['city_filled'] = X['province'] + '-' + X['county'] + '-' + X['city']
        X['area_filled'] = X['province'] + '-' + X['county'] + '-' + X['city'] + '-' + X['area']
        X['neighborhood_filled'] = X['province'] + '-' + X['county'] + '-' + X['city'] + '-' + X['area'] + '-' + X['neighborhood']
        city_price_sqrm = X.groupby('city_filled', as_index=False).apply(lambda x: pd.Series({'location_encoded':x['price'].sum() / x['sqrm'].sum()}))
        city_count = X.groupby('city_filled')['province'].count()
        city_joined = pd.merge(city_price_sqrm, city_count, on='city_filled')
        area_price_sqrm = X.groupby('area_filled', as_index=False).apply(lambda x: pd.Series({'location_encoded':x['price'].sum() / x['sqrm'].sum()}))
        area_count = X.groupby('area_filled')['province'].count()
        area_joined = pd.merge(area_price_sqrm, area_count, on='area_filled')
        area_joined = area_joined.drop(area_joined[area_joined.province < 100].index)
        neighborhood_price_sqrm = X.groupby('neighborhood_filled', as_index=False).apply(lambda x: pd.Series({'location_encoded':x['price'].sum() / x['sqrm'].sum()}))
        neighborhood_count = X.groupby('neighborhood_filled')['province'].count()
        neighborhood_joined = pd.merge(neighborhood_price_sqrm, neighborhood_count, on='neighborhood_filled')
        neighborhood_joined = neighborhood_joined.drop(neighborhood_joined[neighborhood_joined.province < 100].index)
        dict_city = city_joined.drop(columns='province').set_index('city_filled')['location_encoded'].to_dict()
        dict_area = area_joined.drop(columns='province').set_index('area_filled')['location_encoded'].to_dict()
        dict_neighborhood = neighborhood_joined.drop(columns='province').set_index('neighborhood_filled')['location_encoded'].to_dict()
        X['location_encoded'] = X['neighborhood_filled'].map(dict_neighborhood)
        X['location_encoded'] = np.where(X['location_encoded'].isna(), X['area_filled'].map(dict_area), X['location_encoded'])
        X['location_encoded'] = np.where(X['location_encoded'].isna(), X['city_filled'].map(dict_city), X['location_encoded'])
        X=X.drop(labels=['province', 'county', 'city', 'area', 'neighborhood', 'city_filled', 'area_filled', 'neighborhood_filled'], axis=1)
        return X
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)