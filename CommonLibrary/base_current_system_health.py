# basic
import pandas as pd

# class
from abc import ABC, abstractmethod


class BaseCurrentSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        """
        데이터 초기화
        
        Args: 
          data: 입력 데이터프레임
        """
        self.data = data

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

