# basic
import pandas as pd

# type hiting
from typing import Tuple, Union

# class
from abc import ABC, abstractmethod

# module
from .base_rate_change_manager import DataUtility


class BaseFmuSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame, ship_id: str) -> None:
        """
        데이터 초기화
        
        Args: 
          data: 입력 데이터프레임
        """
        self.data = data
        self.ship_id = ship_id

    def calculate_group_health_score(self) -> int:
        """
        그룹의 건강 점수를 계산하는 함수.

        Args:
            col (str): 변화율 계산 및 제한 값 확인에 사용할 열 이름 (예: 'CSU', 'STS', 'FMU').

        Returns:
            Union[Tuple[float, float], float]: 조정된 건강 점수와 트렌드 점수의 튜플,
            또는 특정 형식에 따라 단일 점수.
        """
        filtered_data = self.data[self.data['DATA_INDEX']>=30]
        health_score = filtered_data['HEALTH_RATIO'].max()
        
        return health_score
    
    def apply_system_health_algorithms_with_fmu(self, status) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        FMU 건강도 알고리즘을 적용하는 함수.

        Args:
            status (bool): 결과를 반환할지 여부를 결정하는 플래그. 
                          True[model]일 경우 반환값 없음, False일 경우 데이터프레임 반환.

        Returns:
            Union[None, Tuple[pd.DataFrame, pd.DataFrame]]: 
                status가 False인 경우, 처리된 데이터프레임(`self.data`)과 그룹 데이터프레임(`self.group`)의 튜플.
        """
        self.refine_frames()
        self.generate_health_score('FMU')
        self._col_return()
        self.apply_system_health_statistics_with_fmu()
        
        if not status:
            return self.data, self.group
        
    def generate_health_score(self, col: str) -> None:
        """
        시스템 건강 점수를 생성하는 함수.

        Args:
            data: 입력 데이터프레임
            col: 변화율 계산에 사용할 열 이름

        Returns: 
            pd.DataFrame: 건강 점수가 포함된 데이터프레임
        """
        self.data['STANDARDIZE_FMU'] = self.normalize_series(self.data[[col]])
        self.data['THRESHOLD'] = 1.96
        self.data['STANDARDIZE_FMU'] = self.data['STANDARDIZE_FMU'].abs()
        self.data['HEALTH_RATIO'] = abs( self.data['STANDARDIZE_FMU'] /  self.data['THRESHOLD']) * 30
        self.data = DataUtility.generate_rolling_mean(self.data, col, window=5)
        self._about_col_score_return()
        self.data['HEALTH_RATIO'] = self.data['HEALTH_RATIO'].apply(lambda x: min(100, x))
    
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