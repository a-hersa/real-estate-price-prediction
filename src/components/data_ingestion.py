import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize Data Ingestion Configuration
# @dataclass
# class DataIngestionConfig:
#     train_data_path:str=os.path.join('artifacts','train.csv')
#     test_data_path:str=os.path.join('artifacts','test.csv')
#     raw_data_path:str=os.path.join('artifacts','raw.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self, deal_type):
        self.deal_type = deal_type
        # self.ingestion_config=DataIngestionConfig()
        self.train_data_path:str=os.path.join('artifacts',f'{self.deal_type}_train.csv')
        self.test_data_path:str=os.path.join('artifacts',f'{self.deal_type}_test.csv')
        self.raw_data_path:str=os.path.join('artifacts','raw.csv')

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method Starts')
        try:
            # Reading original csv
            df=pd.read_csv(os.path.join('notebooks','data','realestate.csv'))
            logging.info('Dataset read as pandas Dataframe')

            # Saving a copy as rawdata
            os.makedirs(os.path.dirname(self.raw_data_path),exist_ok=True)
            df.to_csv(self.raw_data_path,index=False)

            # Selecting only the desired deal_type
            df = df[df['type']==self.deal_type]

            # Train-test split
            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=30)

            # Saving Traing-test split
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.train_data_path,
                self.test_data_path
            )


        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        

# test Data Ingestion
if __name__ == '__main__':
    obj = DataIngestion('alquiler')
    train_data, test_data = obj.initiate_data_ingestion()