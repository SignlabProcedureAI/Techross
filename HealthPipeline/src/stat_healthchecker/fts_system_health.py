import pandas as pd
from CommonLibrary import BaseFtsSystemHealth
import os
import numpy as np
from stat_dataline import load_database
from rate_change_manager import RateChangeProcessor
from sklearn.base import BaseEstimator
import pickle

class SimpleFtsSystemHealth(BaseFtsSystemHealth):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def refine_frames(self):
        """
        데이터 프레임 정제
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_fts(self) -> None:
        """ 
        그룹 통계 함수 적용
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        score, trend_score = self.calculate_group_health_score('FTS')
        self.group['HEALTH_SCORE'] = score
        self.group.reset_index(drop=True)
        self.group = self.group[
        [
         'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FTS','HEALTH_SCORE'
        ]
                ]

    def apply_calculating_rate_change(self) -> None:
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'FTS')


    def _col_return(self) -> pd.DataFrame:
        position_columns = [
                 'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FTS','DIFF',
                      'THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        return adjusted_score