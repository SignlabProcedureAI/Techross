import pandas as pd
from CommonLibrary import BaseCurrentSystemHealth
import os
import numpy as np
from stat_dataline import load_database
from rate_change_manager import RateChangeProcessor
from sklearn.base import BaseEstimator
import pickle

class SimpleCurrentSystemHealth(BaseCurrentSystemHealth):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)

    def refine_frames(self):
        """
        데이터 프레임 정제
        """
        columns = [
                    'SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE','START_TIME','END_TIME','RUNNING_TIME'
                    ]
        self.data = self.data[columns]

    def apply_system_health_statistics_with_current(self) -> None:
        """ 
        그룹 통계 함수 적용
        """
        self.data = self.data[self.data['DATA_INDEX'] >=30]
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()['ELECTRODE_EFFICIENCY']
        self.group.assign(
                    START_TIME=self.start_date,
                    END_TIME=self.end_date,
                    RUNNING_TIME=self.running_time,
                    OP_TYPE=self.op_type
                    ).reset_index(drop=True)
        load_database('signlab', 'tc_ai_current_system_health_group', self.group)

    def apply_calculating_rate_change(self) -> None:
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'CSU')


    def _col_return(self) -> pd.DataFrame:
        position_columns = [
                  'SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','CSU','DIFF',
                      'THRESHOLD','HEALTH_RATIO','HEALTH_TREND'
                            ]
        self.data = self.data[position_columns]          

    def _format_return(self, adjusted_score: float, trend_score: float) -> float:
        return adjusted_score
    
    def apply_system_health_algorithms_with_current(self):
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

        self.group = self.apply_system_health_statistics_with_current()
        self.data = self.data[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','ELECTRODE_EFFICIENCY']]

        # 9. 자동 적재
        # load_database('signlab','tc_ai_current_system_health', sensor_data)

        return self.data, self.group