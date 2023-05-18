import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_selector as selector

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from src.utils import RealEstatePreprocessor

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self, deal_type):
        # self.data_transformation_config=DataTransformationConfig()
        self.deal_type = deal_type
        self.preprocessor_obj_file_path=os.path.join('artifacts',f'{self.deal_type}_preprocessor.pkl')

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            logging.info('Pipeline Initiated')

            # Cleaning pipeline
            estate_pipeline = Pipeline(
                steps=[
                    ('real_estate_prep', RealEstatePreprocessor())
                ]
            )

            ## Numerical pipline that scales
            num_pipeline = Pipeline(
                steps=[
                ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline. We don't scale if we are going to One Hot Encode
            cat_pipeline = Pipeline(
                steps=[
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, selector(dtype_exclude=object)),
                ('cat_pipeline', cat_pipeline, selector(dtype_include=object))
            ])

            pipe = Pipeline(
                steps=[
                    ('estate_pipeline', estate_pipeline),
                    ('preprocessor',preprocessor)
                ]
            )
            
            logging.info('Pipeline Completed')

            return pipe

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)           

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            # drop_columns = [target_column_name]

            # input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            train_df = preprocessing_obj[0].transform(train_df)
            test_df = preprocessing_obj[0].transform(test_df)

            X_train = train_df.drop(labels=['price'],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(labels=['price'],axis=1)
            y_test = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing datasets.")

            ## Transformating using preprocessor obj
            X_train=preprocessing_obj[1].fit_transform(X_train)
            X_test=preprocessing_obj[1].transform(X_test)
            
            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            save_object(

                file_path=self.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Exception ocurred in the initiate_data_transformation')
            raise CustomException(e,sys)
        
# test Data Transformation
if __name__ == '__main__':
    obj2 = DataTransformation('alquiler')
    train_arr, test_rr, _ = obj2.initiate_data_transformation(
        os.path.join('artifacts','alquiler_train.csv'),
        os.path.join('artifacts','alquiler_test.csv')
    )