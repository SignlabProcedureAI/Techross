import pandas as pd
import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod
import pickle
from typing import Tuple, Union
from base_rate_change_manager import DataUtility
import os
import joblib

class BaseFmuSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame, ship_id: str) -> None:
        """
        데이터 초기화
        
        :param data: 입력 데이터프레임
        """
        self.data = data
        self.ship_id = ship_id

    def calculate_group_health_score(self) -> int:
            """ 해당 센서 추세를 이용한 건강도 점수 반영
            """
            filtered_data = self.data[self.data['DATA_INDEX']>=30]
            health_score = filtered_data['HEALTH_RATIO'].max()
            
            return health_score
    
    def apply_system_health_algorithms_with_fts(self, status) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """ 
        FTS 건강도 알고리즘 적용 
        """
        self.refine_frames()
        self.generate_health_score('FMU', self.ship_id)
        self._col_return()
        self.apply_system_health_statistics_with_fts()
        
        if status:
            pass
        else: 
            return self.data, self.group

    def generate_health_score(self, col: str) -> None:
        """
        시스템 건강 점수를 생성하는 함수.

        Args:
            col: 변화율 계산에 사용할 열 이름
        """
        self.data['STANDARDIZE_FMU'] = self.normalize_series(self[[col]], self.ship_id)
        self.data['THRESHOLD'] = 1.96
        self.data['STANDARDIZE_FMU'] = self.data['STANDARDIZE_FMU'].abs()
        self.data['HEALTH_RATIO'] = abs( self.data['STANDARDIZE_FMU'] /  self.data['THRESHOLD']) * 30
        self.data = DataUtility.generate_rolling_mean(self.data, col, window=5)
        self._about_col_score_return()
        self.data['HEALTH_RATIO'] = self.data['HEALTH_RATIO'].apply(lambda x: min(100, x))

    @abstractmethod
    def apply_calculating_rate_change(self, col: str):
        """ calculating_rate_change 적용 메소드 (자식 클래스에서 구현 필요)
        """
        pass 
    
    @abstractmethod
    def refine_frames(self):
        """
        데이터 프레임 정제
        """
        pass

    @abstractmethod
    def _format_return(self, adjusted_score: float, trend_score: float):
        """
        반환값 형식을 정의합니다 (자식 클래스에서 구현 필요).

        Args:
        - autocorr: 자기상관 배열
        Returns: 
        - 정의된 반환값
        """
        pass 
    
    @abstractmethod
    def _col_return(self):
        """
        반환값 형식을 정의합니다 (자식 클래스에서 구현 필요)
        """
        pass

    @abstractmethod
    def normalize_series(self, data_series: pd.Series) -> pd.DataFrame:
      pass

    @abstractmethod
    def _about_col_score_return(self):
        pass