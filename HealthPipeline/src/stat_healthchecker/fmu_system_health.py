
# basic
import pandas as pd
import os
import joblib

# type hiting
from typing import Tuple
from sklearn.base import BaseEstimator

# scaler
from sklearn.preprocessing import StandardScaler

# module
from CommonLibrary import BaseCsuSystemHealth
from rate_change_manager import RateChangeProcessor

class SimpleFmuSystemHealth(BaseCsuSystemHealth):
    def refine_frames(self):
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO'
                    ]
        self.data = self.data[columns]

    def create_normalized_series(self, data_series: pd.Series) -> Tuple[pd.DataFrame, BaseEstimator] :
        """
        데이터를 표준화하는 함수.

        Args:
            data_series (pd.Series): 표준화를 수행할 데이터 시리즈.

        Returns:
            Tuple[pd.DataFrame, BaseEstimator]: 
                - 표준화된 데이터 (pd.DataFrame).
                - 데이터에 적합된 스케일러 객체 (BaseEstimator).

        Notes:
            - `StandardScaler`를 사용하여 데이터의 평균을 0, 표준 편차를 1로 조정합니다.
            - 반환된 스케일러 객체를 저장하거나 재사용할 수 있습니다.
        """ 
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data_series)
    
        return standardized_data, scaler
    
    def normalize_series(self, data_series: pd.Series) -> pd.Series:
        """
        데이터를 표준화하는 함수.

        Args:
            data_series (pd.Series): 표준화를 수행할 데이터 시리즈.

        Returns:
            pd.Series: 표준화된 데이터 시리즈.

        Notes:
            - 사전에 저장된 스케일러를 사용해 표준화를 수행합니다.
            - 스케일러 파일이 없을 경우 새로운 스케일러를 생성하고 저장합니다.
            - 스케일러 파일 경로는 `self.ship_id`를 기준으로 결정됩니다.
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
        FMU와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
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
        """
        FMU 열에 대한 변화율을 계산하여 데이터에 적용하는 함수. 
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'FMU')


    def _col_return(self) -> pd.DataFrame:
        """
        필요한 열만 선택하여 반환하는 함수. 
        """
        position_columns = [
                  'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FMU',
                      'STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        """
        포멧 기준 조정된 점수를 반환하는 함수.
        """
        return adjusted_score
    
    def _about_col_score_return(self):
        """
        건강도 점수에 반영하는 변수 반환 
        """
        position_columns = [
                 'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
                'STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                    ]
        self.data = self.data[position_columns]  