# basic
import pandas as pd

# module
from CommonLibrary import BaseCsuSystemHealth
from rate_change_manager import RateChangeProcessor

class SimpleCsuSystemHealth(BaseCsuSystemHealth):
    def refine_frames(self):
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_csu(self) -> None:
        """ 
        CSU와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
        score, trend_score = self.calculate_group_health_score('CSU')
        self.group['HEALTH_SCORE'] = score
        self.group.reset_index(drop=True)
        self.group = self.group[
        [
          'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','HEALTH_SCORE'
        ]
                ]

    def apply_calculating_rate_change(self) -> None:
        """
        CSU 열에 대한 변화율을 계산하여 데이터에 적용하는 함수. 
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'CSU')


    def _col_return(self) -> pd.DataFrame:
        """
        필요한 열만 선택하여 반환하는 함수. 
        """
        position_columns = [
                  'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','CSU','DIFF',
                      'THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        """
        포멧 기준 조정된 점수를 반환하는 함수.
        """
        return adjusted_score