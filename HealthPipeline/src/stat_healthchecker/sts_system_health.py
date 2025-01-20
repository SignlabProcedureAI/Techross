import pandas as pd
from CommonLibrary import BaseStsSystemHealth
import os
import numpy as np
from stat_dataline import load_database
from rate_change_manager import RateChangeProcessor
from sklearn.base import BaseEstimator
import pickle

class SimpleStsSystemHealth(BaseStsSystemHealth):
    def refine_frames(self):
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_sts(self) -> None:
        """ 
        STS와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        score, trend_score = self.calculate_group_health_score('STS')
        self.group['HEALTH_SCORE'] = score
        self.group.reset_index(drop=True)
        self.group = self.group[
        [
          'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','STS','DIFF','HEALTH_SCORE'
        ]
                ]

    def apply_calculating_rate_change(self) -> None:
        """
        STS 열에 대한 변화율을 계산하여 데이터에 적용하는 함수. 
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'STS')


    def _col_return(self) -> pd.DataFrame:
        """
        필요한 열만 선택하여 반환하는 함수. 
        """
        position_columns = [
                 'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','STS','DIFF',
                'THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        """
        포멧 기준 조정된 점수를 반환하는 함수
        """
        return adjusted_score