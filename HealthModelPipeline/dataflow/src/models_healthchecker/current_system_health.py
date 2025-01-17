import pandas as pd
from CommonLibrary import BaseCurrentSystemHealth
import os
import numpy as np
from models_dataline import load_database
from rate_change_manager import RateChangeProcessor
from sklearn.base import BaseEstimator
from typing import Tuple
import pickle

class ModelCurrentystemHealth(BaseCurrentSystemHealth):
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

    def apply_system_health_statistics_with_current(self) -> None:
        """ 
        그룹 통계 함수 적용
        """
        self.data = self.data[self.data['DATA_INDEX'] >=30]
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        [
            'CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY'
        ]
        self.group.assign(
                    START_TIME=self.start_date,
                    END_TIME=self.end_date,
                    RUNNING_TIME=self.running_time,
                    OP_TYPE=self.op_type
                    ).reset_index(drop=True)
        self.group = self.group[
            [
            'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','START_TIME','END_TIME',
            'RUNNING_TIME','CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY'
            ]
                ]
        load_database('ecs_test','tc_ai_electrode_group_v1.1.0', '200', self.group)

        self.predict_stats_val()
        self.group = self.group[
            [
                'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','HEALTH_SCORE','PRED','START_TIME','END_TIME','RUNNING_TIME'
            ]
            ]
        self.group = self.catorize_health_score()
        self.group = self.group.rename({'ELECTRODE_EFFICIENCY':'ACTUAL'}, axis=1)
        self.group['ACTUAL'] = np.round(self.group['ACTUAL'],2)
        self.group['PRED'] = np.round(self.group['PRED'],2)
        load_database('signlab','tc_ai_electrode_model_group', 'release', self.group)

    def predict_stats_val(self) -> None:
        csu_model_relative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models_model/ecu_model_v2.0.0')
        csu_model_path = os.path.abspath(csu_model_relative_path)
        model = self.load_model_from_pickle(csu_model_path)

        X = self.group[['CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT']]
        self.group['PRED'] =  model.predict(X)                      

    def apply_system_health_algorithms_with_current(self):
        """ CURRENT 알고리즘 적용
        Args : 
          선박 이름, 오퍼레이션 번호, 섹션 번호
    
        Returns :
          오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
        """
        self.refine_frames(self.data)
        self.calculate_generalization_value()
        self.calculate_minus_value()

        position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX',
                            'CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY',
                            'START_TIME','END_TIME','RUNNING_TIME']                      
        self.data = self.data[position_columns]                                            
        self.data.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX',
                             'CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT',
                             'ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']
        self.data = self.data[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX',
                               'CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY',
                               'START_TIME','END_TIME','RUNNING_TIME']]
        self.group = self.apply_system_health_statistics_with_current()

        return self.data, self.group

    def catorize_health_score(self) -> None:
        self.data['DEFECT_RISK_CATEGORY'] = 0
        self.data.loc[self.data['ELECTRODE_EFFICIENCY']>=-16, 'RISK'] = 'NORMAL'
        self.data.loc[(self.data['ELECTRODE_EFFICIENCY']<-16) & (self.data['ELECTRODE_EFFICIENCY']>=-40), 'RISK'] = 'WARNING'
        self.data.loc[(self.data['ELECTRODE_EFFICIENCY']<-40) & (self.data['ELECTRODE_EFFICIENCY']>=-90), 'RISK'] = 'RISK'
        self.data.loc[self.data['ELECTRODE_EFFICIENCY']<-90, 'RISK'] = 'DEFECT'

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

