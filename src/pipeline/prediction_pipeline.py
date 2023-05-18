import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self,deal_type):
        self.deal_type = deal_type

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts',f'{self.deal_type}_preprocessor.pkl')
            model_path = os.path.join('artifacts',f'{self.deal_type}_model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info('Exception occurred in prediction')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(
            self,
            deal_type:str,
            parking:int,
            rooms:int,
            sqrm:int,
            floor:int,
            surface:int,
            elevator:int,
            property_type:str,
            location_encoded:int):
        
        self.deal_type = deal_type
        self.parking = parking
        self.rooms = rooms
        self.sqrm = sqrm
        self.floor = floor
        self.surface = surface
        self.elevator = elevator
        self.property_type = property_type
        self.location_encoded = location_encoded
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'deal_type':[self.deal_type],
                'parking':[self.parking],
                'rooms':[self.rooms],
                'sqrm':[self.sqrm],
                'floor':[self.floor],
                'surface':[self.surface],
                'elevator':[self.elevator],
                'property_type':[self.property_type],
                'location_encoded':[self.location_encoded]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame generated')
            return df
        
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e,sys)