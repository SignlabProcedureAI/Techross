# basic
import pandas as pd
import numpy as np

# type hinting
from typing import Union, Tuple

# module
from base import BaseCurrentSystemHealth
from stat_dataline import load_database
from .rate_change_manager import RateChangeProcessor

class SimpleCurrentSystemHealth(BaseCurrentSystemHealth):
    def refine_frames(self):
        """
        데이터 프레임에서 필요한 열만 선택하여 정제하는 함수
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE','START_TIME','END_TIME','RUNNING_TIME'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_current(self) -> None:
        """ 
        CUREENT와 관련된 그룹 통계와 건강 점수를 계산하여 데이터 프레임에 적용하는 함수
        """
        self.data = self.data[self.data['DATA_INDEX'] >=30]
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()['ELECTRODE_EFFICIENCY'].to_frame().reset_index()
        self.group = self.group.assign(
                    START_TIME=self.start_date,
                    END_TIME=self.end_date,
                    RUNNING_TIME=self.running_time,
                    OP_TYPE=self.op_type
                    )
        # load_database('signlab', 'tc_ai_current_system_health_group', self.group)
        load_database.load_database('ecs_test', 'test_tc_ai_current_system_health_group', self.group)

    def apply_calculating_rate_change(self) -> None:
        """
        CURRENT 열에 대한 변화율을 계산하여 데이터에 적용하는 함수. 
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'CURRENT')


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
    
    def apply_system_health_algorithms_with_current(self, status: bool) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """ CURRENT 건강도 알고리즘 적용 
        Args: 
        선박 이름, 오퍼레이션 번호, 섹션 번호
        
        Returns: 
        오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
        """
        self.refine_frames()
        self.calculate_generalization_value()
        self.calculate_minus_value()
        

        position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX',
                            'ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']                      
        self.data = self.data[position_columns]                                               
        self.data.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']
        
        self.data['ELECTRODE_EFFICIENCY'] = np.round(self.data['ELECTRODE_EFFICIENCY'],2)

        self.apply_system_health_statistics_with_current()
        self.data = self.data[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','ELECTRODE_EFFICIENCY']]

        # 9. 자동 적재
        # load_database('signlab','tc_ai_current_system_health', sensor_data)

        if not status:
            return self.data, self.group