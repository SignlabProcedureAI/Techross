# basic
import pandas as pd

# module
from base import BaseFtsSystemHealth
from .rate_change_manager import RateChangeProcessor

class SimpleFtsSystemHealth(BaseFtsSystemHealth):
    def refine_frames(self):
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_fts(self) -> None:
        """ 
        FTS와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        score = self.calculate_group_health_score('FTS')
        self.group['HEALTH_SCORE'] = score
        self.group = self.group.reset_index()
        self.group = self.group[
        [
         'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FTS','HEALTH_SCORE'
        ]
                ]

    def apply_calculating_rate_change(self) -> None:
        """
        FTS 열에 대한 변화율을 계산하여 데이터에 적용하는 함수. 
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'FTS')


    def _col_return(self) -> pd.DataFrame:
        """
        필요한 열만 선택하여 반환하는 함수. 
        """
        position_columns = [
                 'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FTS','DIFF',
                      'THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        """
        포멧 기준 조정된 점수를 반환하는 함수
        """
        return adjusted_score
    
    def _about_score_col_return(self):
        self.data.columns = [
            'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX',
            'CSU','STS','FTS','FMU','CURRENT','TRO',
            'FTS_Ratio','THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                    ]