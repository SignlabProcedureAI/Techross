from abc import ABC, abstractmethod
import pandas as pd
import os
import numpy as np
import pickle
from .base_rate_change_manager import DataUtility 

class BaseFaultAlgorithm(ABC):
    """
    TRO 센서 불량 탐지를 위한 베이스 클래스.
    공통 기능과 확장 가능한 구조를 제공합니다.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Args:
         data (pd.DataFrame): 처리할 데이터프레임
        """
        self.data['DATA_TIME'] = pd.to_datetime(self.data['DATA_TIME'])
        self.data['START_TIME'] = pd.to_datetime(self.data['START_TIME'])
        self.data['END_TIME'] = pd.to_datetime(self.data['END_TIME'])

        self.start_date = self.data.iloc[0]['START_TIME']
        self.end_date = self.data.iloc[0]['END_TIME']
        self.running_time = self.data.iloc[0]['RUNNING_TIME']
        self.op_type = self.data.iloc[0]['OP_TYPE']


    def refine_frames(self) -> None:
        """
        데이터에서 필요한 열만 선택하여 정제
        """
        columns = ['SHIP_ID', 'OP_INDEX', 'SECTION', 'OP_TYPE', 'DATA_TIME', 'DATA_INDEX',
                   'CSU', 'STS', 'FTS', 'FMU', 'CURRENT', 'TRO',
                   'START_TIME', 'END_TIME', 'RUNNING_TIME'
                   ]
        return self.data[columns]
    
    def predict_tro_val(self) -> None:
        """
        예측 모델을 사용하여 데이터를 처리합니다.
        """
        columns = ['CSU', 'STS', 'FTS', 'FMU', 'CURRENT']
        tro_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models_model/tro_model')
        tro_model = self.load_model_from_pickle(tro_model_path)

        independent_vars = self.data[columns]
        self.data['pred'] = tro_model.predict(independent_vars)

    def update_tro_condition(self) -> None:
        """
        TRO 조건 부여 로직
        """
        last_10_values = self.data['DATA_INDEX'].iloc[-10:].values

        self.data.loc[self.data['TRO'] >= 8, 'STEEP_LABEL'] = 0
        self.data.loc[(self.data['DATE_INDEX']>10) & (self.data['DATA_INDEX'].isin(last_10_values)), 'STEEP_LABEL'] = 0
        self.data.loc[(self.data['DATE_INDEX']>10) & (self.data['DATA_INDEX'].isin(last_10_values)), 'OUT_OF_WATER_STEEP'] = 0  
      
    
    def give_tro_out_of_water_condition(self) -> None:
        """
        TRO Out of Water 조건 부여 로직.
        Args:
         데이터프레임

        Returns:  
         조건이 부여된 데이터프레임
        """
        self.data['OUT_OF_WATER_STEEP'] = 0
        self.data.loc[(self.data['TRO'] <= 1) & (self.data['STEEP_LABEL'] == 1), 'OUT_OF_WATER_STEEP'] = 1
        self.data.loc[(self.data['TRO'] <= 1) & (self.data['STEEP_LABEL'] == 1), 'STEEP_LABEL'] = 0


    def apply_automation_labeling(self) -> None:
        """ 자동화 라벨링 알고리즘을 적용
        """
        # Steep Label 함수 적용
        self.data['steep_label'] = self.data.apply(lambda x : DataUtility.classify_decline_steep_label(x['TRO_Ratio'],x['pred_Ratio']),axis=1)
        
        # 이전 값을 나타내는 열 추가
        self.data['Previous_TRO_NEG_COUNT'] = self.data['TRO_NEG_COUNT'].shift(1)
        
        # shif(1)에 의한 결측치 제거
        self.data.dropna(inplace=True)
        self.data['slowly_label'] = self.data.apply(lambda x :DataUtility.classify_decline_slowly_label(x['Previous_TRO_NEG_COUNT'],x['TRO_NEG_COUNT']),axis=1)
        self.data = self.data.drop(columns='Previous_TRO_NEG_COUNT')

    @abstractmethod
    def apply_fault_label_statistics(self) -> pd.DataFrame:
        """
        그룹화된 데이터의 통계를 계산하는 추상 메서드 (하위 클래스에서 구현)
        """
        pass

    @abstractmethod
    def apply_tro_labeling(self) -> pd.DataFrame:
        """
        TRO 관련 자동화 알고리즘 적용 함수 (하위 클래스에서 구현)
        """
        pass

    @abstractmethod
    def apply_tro_fault_detector(self):
        """ 
        TRO 관련 라벨링 알고리즘을 적용 함수 (하위 클래스에서 구현) 
        """
        pass

