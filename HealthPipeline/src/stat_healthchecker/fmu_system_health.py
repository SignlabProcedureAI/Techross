import pandas as pd
from CommonLibrary import BaseCsuSystemHealth
import os
import numpy as np
from stat_dataline import load_database
from rate_change_manager import RateChangeProcessor
from sklearn.base import BaseEstimator
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple

class SimpleFmuSystemHealth(BaseCsuSystemHealth):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def refine_frames(self):
        """
        데이터 프레임 정제
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO'
                    ]
        self.data = self.data[columns]

    def create_normalized_series(self, data_series: pd.Series) -> Tuple[pd.DataFrame, BaseEstimator] :
        """ 표준화 함수
        """   
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data_series)
    
        return standardized_data, scaler
    
    def normalize_series(self, data_series: pd.Series) -> pd.Series:
        """ 표준화 함수
        """   
        fmu_scaler_path = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(fmu_scaler_path, '../../data/fmu_standard_scaler'))
        ship_scaler_path = os.path.join(data_dir, f'{self.ship_id}_scaler.joblib')

        if os.path.exists(ship_scaler_path):
            scaler =  joblib.load(fr'{data_dir}\{self.ship_id}_scaler.joblib')
            standardized_data = scaler.transform(data_series)
        else:
            standardized_data, scaler = self.create_normalized_series(data_series)
            joblib.dump(scaler, ship_scaler_path) 

        return standardized_data
        

    def apply_system_health_statistics_with_fmu(self) -> None:
        """ 
        그룹 통계 함수 적용
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        score, trend_score = self.calculate_group_health_score('FMU')
        self.group['HEALTH_SCORE'] = score
        self.group.reset_index(drop=True)
        self.group = self.group[
        [
          'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FMU','STANDARDIZE_FMU','HEALTH_SCORE'
        ]
                ]

    def apply_calculating_rate_change(self) -> None:
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'FMU')


    def _col_return(self) -> pd.DataFrame:
        position_columns = [
                  'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FMU',
                      'STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        return adjusted_score
    
    def _about_col_score_return(self):
        position_columns = [
                 'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
                'STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                    ]
        self.data = self.data[position_columns]  