
# basic
import pandas as pd
import os
import numpy as np
import pickle

# type hiting
from sklearn.base import BaseEstimator
from typing import Union, Tuple

# module
from base import BaseCurrentSystemHealth
from models_dataline import load_database

class ModelCurrentystemHealth(BaseCurrentSystemHealth):
    def refine_frames(self) -> None:
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE',
                    'START_TIME','END_TIME','RUNNING_TIME'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_current(self) -> None:
        """ 
        CURRENT와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
        """
        self.data = self.data[self.data['DATA_INDEX'] >=30]
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        [
            'CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY'
        ]
        self.group = self.group.assign(
                    START_TIME=self.start_date,
                    END_TIME=self.end_date,
                    RUNNING_TIME=self.running_time,
                    OP_TYPE=self.op_type
                    ).reset_index()
        self.group = self.group[
            [
            'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','START_TIME','END_TIME',
            'RUNNING_TIME','CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY'
            ]
                ]
        load_database('ecs_test','test_tc_ai_electrode_group_v1.1.0', '200', self.group)

        self.predict_stats_val()
        self.group = self.group[
            [
                'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','ELECTRODE_EFFICIENCY','PRED','START_TIME','END_TIME','RUNNING_TIME'
            ]
            ]
        self.catorize_health_score()
        self.group = self.group.rename({'ELECTRODE_EFFICIENCY':'ACTUAL'}, axis=1)
        self.group['ACTUAL'] = np.round(self.group['ACTUAL'],2)
        self.group['PRED'] = np.round(self.group['PRED'],2)
        # load_database('signlab','tc_ai_electrode_model_group', 'release', self.group)
        load_database('ecs_test','test_tc_ai_electrode_model_group', '200', self.group)

    def predict_stats_val(self) -> None:
        """
        통계 데이터를 사용하여 예측 값을 계산하는 함수.

        Description:
            - 저장된 CURRENT 모델을 로드하여 데이터의 통계 값을 기반으로 예측을 수행합니다.
            - 예측 결과는 그룹 데이터 프레임(self.group)에 'PRED' 열로 추가됩니다.
        """
        ecu_model_relative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models_model/ecu_model_v2.0.0')
        ecu_model_path = os.path.abspath(ecu_model_relative_path)
        model = self.load_model_from_pickle(ecu_model_path)

        X = self.group[['CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT']]
        self.group['PRED'] =  model.predict(X)                      

    def apply_system_health_algorithms_with_current(self, status: bool) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """ CURRENT 알고리즘 적용
        Args : 
          선박 이름, 오퍼레이션 번호, 섹션 번호
    
        Returns :
          오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
        """
        self.refine_frames()
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

        if not status:
            return self.data, self.group

    def catorize_health_score(self) -> None:
        """
        건강 점수를 기반으로 결함 위험 카테고리를 분류하는 함수

        Description:
            - HEALTH_SCORE 값에 따라 각 데이터 포인트를 'NORMAL', 'WARNING', 'RISK', 'DEFECT' 카테고리로 분류합니다
            - 분류 결과는 'RISK' 열에 저장됩니다.
        """
        self.group['DEFECT_RISK_CATEGORY'] = 0
        self.group.loc[self.group['ELECTRODE_EFFICIENCY']>=-16, 'RISK'] = 'NORMAL'
        self.group.loc[(self.group['ELECTRODE_EFFICIENCY']<-16) & (self.group['ELECTRODE_EFFICIENCY']>=-40), 'RISK'] = 'WARNING'
        self.group.loc[(self.group['ELECTRODE_EFFICIENCY']<-40) & (self.group['ELECTRODE_EFFICIENCY']>=-90), 'RISK'] = 'RISK'
        self.group.loc[self.group['ELECTRODE_EFFICIENCY']<-90, 'RISK'] = 'DEFECT'

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

