# basic
import pandas as pd

# class
from abc import ABC, abstractmethod


class BaseCurrentSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        """
        CURRENT 시스템 건강도를 모델링하는 클래스의 초기화 메서드.

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
        
    def calculate_minus_value(self):
        """
        ELECTRODE_EFFICIENCY 값을 계산하는 함수.

        Steps:
            1. GENERALIZATION_EFFICIENCY를 기반으로 ELECTRODE_EFFICIENCY 계산.
            2. ELECTRODE_EFFICIENCY 값이 양수인 경우 0으로 처리.

        Returns:
            None: self.data가 업데이트됩니다.
        """
        self.data['ELECTRODE_EFFICIENCY'] =  - (100 - self.data['GENERALIZATION_EFFICIENCY'])
        self.data['ELECTRODE_EFFICIENCY'] = self.data['ELECTRODE_EFFICIENCY'].apply(lambda x: 0 if x >= 0 else x)

    def calculate_generalization_value(self):
        """
        GENERALIZATION_EFFICIENCY 값을 계산하는 함수.

        Formula:
            GENERALIZATION_EFFICIENCY = 
                (100 * TRO) / ((1.323 * CURRENT) / FMU) * (1 + k * (1 - CSU / 50))
        """
        k = 0.23
        self.data['GENERALIZATION_EFFICIENCY'] = (100*self.data['TRO'])/((1.323*self.data['CURRENT']) / self.data['FMU']) * (1 + k * (1 - self.data['CSU'] / 50))

    
    @abstractmethod
    def refine_frames(self):
        """
        데이터 프레임 정제 함수.
        """
        pass

