# basic
import pandas as pd

# type hiting
from typing import Tuple, Union

# class
from abc import ABC, abstractmethod

# module
from .base_rate_change_manager import DataUtility

class BaseFtsSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        """
        데이터 초기화
        
        Args: 
          data: 입력 데이터프레임
        """
        self.data = data

    def exceed_limit_line(self, col: str) -> int:
        """
        특정 열의 최대값이 제한 값을 초과하는지 확인하는 함수.

        Args:
            col (str): 확인할 열 이름 (예: 'CSU', 'STS', 'FTS').

        Returns:
            int: 최대값이 제한 값을 초과하면 100, 그렇지 않으면 0.
        """
        limit = {
            'CSU': 45,
            'STS' : 40,
            'FTS' : 40
            }
        max_val = self.data[col].max()
        
        if max_val > limit[col]:
            return 100
        else :
            return 0
    
    def calculate_group_health_score(self, col: str) -> Union[Tuple[float,float], float]:
        """
        그룹의 건강 점수를 계산하는 함수.

        Args:
            col (str): 변화율 계산 및 제한 값 확인에 사용할 열 이름 (예: 'CSU', 'STS', 'FTS').

        Returns:
            Union[Tuple[float, float], float]: 조정된 건강 점수와 트렌드 점수의 튜플,
            또는 특정 형식에 따라 단일 점수.
        """
        threshold = {
            'CSU': 0.88,
            'STS' : 1.18,
            'FTS': 1.75
                }
        
        start_value = self.data.iloc[0][col]
        end_value = self.data.iloc[-1][col]

        trend_score = (abs(end_value - start_value) / threshold[col]) * 10
        health_score = self.data['HEALTH_RATIO'].max()
        total_score = health_score + trend_score
        limit_score = self.exceed_limit_line(col)
        adjusted_score = min(100, total_score + limit_score)

        return self._format_return(adjusted_score, trend_score)
    
    def apply_system_health_algorithms_with_fts(self, status) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        FTS 건강도 알고리즘을 적용하는 함수.

        Args:
            status (bool): 결과를 반환할지 여부를 결정하는 플래그. 
                          True[model]일 경우 반환값 없음, False일 경우 데이터프레임 반환.

        Returns:
            Union[None, Tuple[pd.DataFrame, pd.DataFrame]]: 
                status가 False인 경우, 처리된 데이터프레임(`self.data`)과 그룹 데이터프레임(`self.group`)의 튜플.
        """
        self.refine_frames()
        self.generate_health_score('FTS')
        self._col_return()
        self.apply_system_health_statistics_with_fts()
    
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

        # 변화율 계산 (자식 클래스에서 오버라이드)
        self.apply_calculating_rate_change()

        self.data.dropna(inplace=True)
        self.data['THRESHOLD'] = 0.22
        self.data[f'{col}_Ratio'] = self.data[f'{col}_Ratio'].abs()
        self.data['HEALTH_RATIO'] = (self.data[f'{col}_Ratio'] / self.data['THRESHOLD']).abs() * 10
        self.data = DataUtility.generate_rolling_mean(self.data, col, window=5)

        self._about_score_col_return()
        self.data.rename(columns={'FTS_Ratio': 'DIFF'}, inplace=True)
        self.data['HEALTH_RATIO'] = self.data['HEALTH_RATIO'].apply(lambda x: min(100, x))

    @abstractmethod
    def apply_calculating_rate_change(self):
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
    def _about_score_col_return(self):
        """
        반환값 형식을 정의합니다 (자식 클래스에서 구현 필요)
        """
        pass
