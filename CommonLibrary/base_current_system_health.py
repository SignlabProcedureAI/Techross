import pandas as pd
import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod
import pickle
from typing import Tuple, Union
from base_rate_change_manager import DataUtility

class BaseCurrentSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        """
        데이터 초기화
        
        :param data: 입력 데이터프레임
        """
        self.data = data

    def calculate_minus_value(self):
        self.data['ELECTRODE_EFFICIENCY'] =  - (100 - self.data['GENERALIZATION_EFFICIENCY'])
        self.data['ELECTRODE_EFFICIENCY'] = self.data['ELECTRODE_EFFICIENCY'].apply(lambda x: 0 if x >= 0 else x)

    def calculate_generalization_value(self):
        k = 0.23
        self.data['GENERALIZATION_EFFICIENCY'] = (100*self.data['TRO'])/((1.323*self.data['CURRENT']) / self.data['FMU']) * (1 + k * (1 - self.data['CSU'] / 50))

    
    @abstractmethod
    def refine_frames(self):
        """
        데이터 프레임 정제
        """
        pass

