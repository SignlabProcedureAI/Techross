# basic
import pandas as pd
import pickle
import joblib
import os
import numpy as np

# type hiting
from sklearn.base import BaseEstimator
from typing import Tuple

# module
from CommonLibrary import BaseFmuSystemHealth
from models_dataline import load_database
from rate_change_manager import RateChangeProcessor

class ModelFmuSystemHealth(BaseFmuSystemHealth):
    def __init__(self, data: pd.DataFrame):
        """
        FMU 시스템 건강도를 모델링하는 클래스의 초기화 메서드.

        Args:
            data (pd.DataFrame): 초기화에 사용할 입력 데이터프레임.

        Attributes:
            start_date (datetime): 데이터의 첫 행에서 추출한 시작 시간.
            end_date (datetime): 데이터의 첫 행에서 추출한 종료 시간.
            running_time (float): 데이터의 첫 행에서 추출한 실행 시간.
            op_type (str): 데이터의 첫 행에서 추출한 운영 유형.
        """
        self.data = data

        for col in ['DATA_TIME', 'START_TIME', 'END_TIME']:
            self.data[col] = pd.to_datetime(self.data[col])

        first_row = self.data.iloc[0]
        self.start_date = first_row['START_TIME']
        self.end_date = first_row['END_TIME']
        self.running_time = first_row['RUNNING_TIME']
        self.op_type = first_row['OP_TYPE']

    def refine_frames(self) -> None:
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE',
                    'START_TIME','END_TIME','RUNNING_TIME'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_fmu(self) -> None:
        """ 
        FMU와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
        """
        self.data = self.data[
            [
            'SHIP_ID','OP_INDEX','DATA_INDEX','SECTION','CSU','STS','FTS','CURRENT','TRO','FMU','STANDARDIZE_FMU','THRESHOLD',
           'HEALTH_RATIO','HEALTH_TREND'
            ]
           ]
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg
        (
            {
            'DATA_INDEX':'mean','CSU':'mean','STS':'mean','FTS':'mean','CURRENT':'mean','TRO':'mean',
            'FMU':['min','mean','max'],'STANDARDIZE_FMU':['min','mean','max'],
            'THRESHOLD':'mean','HEALTH_RATIO':'mean','HEALTH_TREND':'mean'
            }
        )
        # 다중 인덱스된 컬럼을 단일 레벨로 평탄화
        self.group.columns = ['_'.join(col) for col in self.group.columns]
        self.group.columns = [
                            'DATA_INDEX','CSU','STS','FTS','CURRENT','TRO','FMU_MIN','FMU_MEAN','FMU_MAX',
                            'STANDARDIZE_FMU_MIN','STANDARDIZE_FMU_MEAN','STANDARDIZE_FMU_MAX','THRESHOLD',
                            'HEALTH_RATIO','HEALTH_TREND'
                            ]
        score, trend_score = self.calculate_group_health_score('FMU')
        self.group.assign(
                    HEALTH_SCORE=score,
                    TREND_SCORE=trend_score,
                    START_TIME=self.start_date,
                    END_TIME=self.end_date,
                    RUNNING_TIME=self.running_time,
                    OP_TYPE=self.op_type
                    ).reset_index(drop=True)
        self.group = self.group[
            [
           'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','CSU','STS','FTS','CURRENT','TRO',
           'FMU_MIN','FMU_MEAN','FMU_MAX','STANDARDIZE_FMU_MIN','STANDARDIZE_FMU_MEAN','STANDARDIZE_FMU_MAX','THRESHOLD',
           'HEALTH_RATIO','HEALTH_TREND','HEALTH_SCORE','START_TIME','END_TIME','RUNNING_TIME'
            ]
                ]
        load_database('ecs_test','tc_ai_fmu_system_health_group_v1.1.0', '200', self.group)

        self.predict_stats_val()
        self.group = self.group[
            [
                'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','HEALTH_SCORE','PRED','START_TIME','END_TIME','RUNNING_TIME'
            ]
            ]
        self.group = self.catorize_health_score()
        self.group = self.group.rename({'HEALTH_SCORE':'ACTUAL'}, axis=1)
        self.group['ACTUAL'] = np.round(self.group['ACTUAL'],2)
        self.group['PRED'] = np.round(self.group['PRED'],2)
        load_database('signlab','tc_ai_fmu_model_system_health_group', 'release', self.group)
    
    def apply_calculating_rate_change(self) -> None:
        """
        FMU 열에 대한 변화율을 계산하여 데이터에 적용하는 함수. 
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'FMU')

    def predict_stats_val(self) -> None:
        """
        통계 데이터를 사용하여 예측 값을 계산하는 함수.

        Description:
            - 저장된 FMU 모델을 로드하여 데이터의 통계 값을 기반으로 예측을 수행합니다.
            - 예측 결과는 그룹 데이터 프레임(self.group)에 'PRED' 열로 추가됩니다.
        """
        fmu_model_relative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models_model/fmu_model_v2.0.0')
        fmu_model_path = os.path.abspath(fmu_model_relative_path)
        model = self.load_model_from_pickle(fmu_model_path)

        X = self.group[['FMU_MIN','STANDARDIZE_FMU_MIN','STANDARDIZE_FMU_MEAN','STANDARDIZE_FMU_MAX']]
        self.group['PRED'] =  model.predict(X)

    def _col_return(self) -> None:
        """
        필요한 열만 선택하여 반환하는 함수. 
        """
        position_columns = [
                   'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','CURRENT','TRO','FMU','STANDARDIZE_FMU',
                    'THRESHOLD','HEALTH_RATIO','HEALTH_TREND','START_TIME','END_TIME','RUNNING_TIME'
                    ]
        self.data = self.data[position_columns]                              

    def _format_return(self, adjusted_score: float, trend_score: float) -> Tuple[float,float]:
        """
        포멧 기준 조정된 점수를 반환하는 함수.
        """
        return adjusted_score, trend_score

    def _about_col_score_return(self):
        """ 건강도 점수에 반영하는 변수 반환 
        """
        position_columns = [
                  'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE', 'VOLTAGE',
                'START_TIME','END_TIME','RUNNING_TIME','STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                    ]
        self.data = self.data[position_columns]   

    def catorize_health_score(self) -> None:
        """
        건강 점수를 기반으로 결함 위험 카테고리를 분류하는 함수

        Description:
            - HEALTH_SCORE 값에 따라 각 데이터 포인트를 'NORMAL', 'WARNING', 'RISK', 'DEFECT' 카테고리로 분류합니다
            - 분류 결과는 'RISK' 열에 저장됩니다.
        """
        self.data['DEFECT_RISK_CATEGORY'] = 0
        self.data.loc[self.data['HEALTH_SCORE']<=23, 'RISK'] = 'NORMAL'
        self.data.loc[(self.data['HEALTH_SCORE']>23) & (self.data['HEALTH_SCORE']<=40), 'RISK'] = 'WARNING'
        self.data.loc[(self.data['HEALTH_SCORE']>40) & (self.data['HEALTH_SCORE']<=80), 'RISK'] = 'RISK'
        self.data.loc[self.data['HEALTH_SCORE']>80, 'RISK'] = 'DEFECT'

    def normalize_series(self, data_series: pd.Series) -> pd.DataFrame:
        """
        데이터를 표준화/정규화하는 함수.

        Args:
            data_series (pd.Series): 정규화를 수행할 데이터 시리즈.

        Returns:
            pd.DataFrame: 정규화된 데이터.

        Notes:
            - 정규화를 위해 사전 저장된 스케일러 파일을 로드합니다.
            - 스케일러 파일 경로는 `self.ship_id`를 기반으로 동적으로 결정됩니다.
        """ 
        fmu_scaler_path = os.path.dirname(os.path.abspath(__file__))
        scaler_dir = os.path.abspath(os.path.join(fmu_scaler_path, '../../../../HealthPipeline/data/fmu_standard_scaler'))
        scaler =  joblib.load(fr'{scaler_dir}\\{self.ship_id}_scaler.joblib')
        standardized_data = scaler.transform(data_series)
        
        return standardized_data
    
    @staticmethod
    def load_model_from_pickle(file_path: str) -> BaseEstimator:
        """
        피클 파일에서 모델을 불러오는 함수.

        Args:
        - file_path: 불러올 피클 파일의 경로

        Returns:
        - model: 불러온 모델 객체
        """
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"모델이 {file_path}에서 성공적으로 불러와졌습니다.")
        return model

