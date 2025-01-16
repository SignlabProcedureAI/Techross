import pandas as pd
from CommonLibrary import BaseCsuSystemHealth
import os
import numpy as np
from models_dataline import load_database
from rate_change_manager import RateChangeProcessor
from sklearn.base import BaseEstimator
from typing import Tuple
import pickle

class ModelCsuSystemHealth(BaseCsuSystemHealth):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

        for col in ['DATA_TIME', 'START_TIME', 'END_TIME']:
            self.data[col] = pd.to_datetime(self.data[col])

        first_row = self.data.iloc[0]
        self.start_date = first_row['START_TIME']
        self.end_date = first_row['END_TIME']
        self.running_time = first_row['RUNNING_TIME']
        self.op_type = first_row['OP_TYPE']

    def refine_frames(self) -> None:
        """
        데이터 프레임 정제
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE',
                    'START_TIME','END_TIME','RUNNING_TIME'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_csu(self) -> None:
        """ 
        그룹 통계 함수 적용
        """
        self.data = self.data[
            [
            'SHIP_ID','OP_INDEX','DATA_INDEX','SECTION','STS','FTS',
             'FMU','CURRENT','TRO','CSU','DIFF','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
            ]
           ]
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg
        (
            {'DATA_INDEX':'mean','STS':'mean','FTS':'mean','FMU':'mean','CURRENT':'mean','TRO':'mean','CSU':['min','mean','max'],
             'DIFF':['min','mean','max'],'THRESHOLD':'mean','HEALTH_RATIO':'mean','HEALTH_TREND':'mean'
            }
        )
        # 다중 인덱스된 컬럼을 단일 레벨로 평탄화
        self.group.columns = ['_'.join(col) for col in self.group.columns]
        self.group.columns = [
                            'DATA_INDEX','STS','FTS','FMU','CURRENT','TRO',
                            'CSU_MIN','CSU_MEAN','CSU_MAX','DIFF_MIN','DIFF_MEAN',
                            'DIFF_MAX','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        score, trend_score = self.calculate_group_health_score('CSU')
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
            'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','STS','FTS','FMU','CURRENT','TRO','CSU_MIN','CSU_MEAN','CSU_MAX',
            'DIFF_MIN','DIFF_MEAN','DIFF_MAX','THRESHOLD','TREND_SCORE','HEALTH_RATIO','HEALTH_TREND','HEALTH_SCORE'
            ,'START_TIME','END_TIME','RUNNING_TIME'
            ]
                ]
        load_database('ecs_test','tc_ai_csu_system_health_group_v1.1.0', '200', self.group)

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
        load_database('signlab','tc_ai_csu_model_system_health_group', 'release', self.group)
    
    def apply_calculating_rate_change(self) -> None:
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'CSU')

    def predict_stats_val(self) -> None:
        csu_model_relative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models_model/csu_model_v2.0.0')
        csu_model_path = os.path.abspath(csu_model_relative_path)
        model = self.load_model_from_pickle(csu_model_path)

        X = self.group[['CSU_MIN', 'CSU_MEAN', 'CSU_MAX', 'DIFF_MIN', 'DIFF_MEAN', 'DIFF_MAX', 'TREND_SCORE']]
        self.group['PRED'] =  model.predict(X)

    def _col_return(self) -> None:
        position_columns = [
                    'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','STS','FTS','FMU','CURRENT','TRO','CSU','DIFF',
                    'THRESHOLD','HEALTH_RATIO','HEALTH_TREND','START_TIME','END_TIME','RUNNING_TIME'
                            ]
        self.data = self.data[position_columns]                              

    def _format_return(self, adjusted_score: float, trend_score: float) -> Tuple[float,float]:
        return adjusted_score, trend_score

    def catorize_health_score(self) -> None:
        self.data['DEFECT_RISK_CATEGORY'] = 0
        self.data.loc[self.data['HEALTH_SCORE']<=14, 'RISK'] = 'NORMAL'
        self.data.loc[(self.data['HEALTH_SCORE']>15) & (self.data['HEALTH_SCORE']<=40), 'RISK'] = 'WARNING'
        self.data.loc[(self.data['HEALTH_SCORE']>40) & (self.data['HEALTH_SCORE']<=90), 'RISK'] = 'RISK'
        self.data.loc[self.data['HEALTH_SCORE']>90, 'RISK'] = 'DEFECT'

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

