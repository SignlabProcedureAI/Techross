import pandas as pd
import numpy as np
from typing import Tuple, Union
from abc import ABC, abstractmethod
import pickle
from typing import Tuple, Union
from base_rate_change_manager import DataUtility

class BaseFtsSystemHealth(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        """
        데이터 초기화
        
        :param data: 입력 데이터프레임
        """
        self.data = data

    def exceed_limit_line(self, col: str) -> int:
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
        FTS 건강도 알고리즘 적용 
        """
        self.refine_frames()
        self.generate_health_score('FTS')
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
            data: 입력 데이터프레임
            col: 변화율 계산에 사용할 열 이름

        Returns: 
            pd.DataFrame: 건강 점수가 포함된 데이터프레임
        """

        # 변화율 계산 (자식 클래스에서 오버라이드)
        self.apply_calculating_rate_change(col)

        self.data.dropna(inplace=True)
        self.data['THRESHOLD'] = 0.22
        self.data[f'{col}_Ratio'] = self.data[f'{col}_Ratio'].abs()
        self.data['HEALTH_RATIO'] = (self.data[f'{col}_Ratio'] / self.data['THRESHOLD']).abs() * 10
        self.data = DataUtility.generate_rolling_mean(self.data, col, window=5)

        self.data.columns = [
           'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
            'FTS_Ratio','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                ]
        self.data.rename(columns={'FTS_Ratio': 'DIFF'}, inplace=True)
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

    def _col_return(self):
        """
        반환값 형식을 정의합니다 (자식 클래스에서 구현 필요)
        """
        pass
