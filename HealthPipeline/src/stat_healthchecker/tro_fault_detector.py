from CommonLibrary import BaseFaultAlgorithm
import pandas as pd
from  stat_dataline import load_database
import os
from rate_change_manager import RateChangeProcessor
from hunting_processor import StatHungting

class TROFaultAlgorithm(BaseFaultAlgorithm):
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): 처리할 데이터프레임
        """ 
        super().__init__(data)

    def apply_tro_labeling(self):
        self.data = RateChangeProcessor.calculating_rate_change(self.data, 'TRO')
        self.data = RateChangeProcessor.calculating_rate_change(self.data, 'pred')

        position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT',
                            'TRO','TRO_Ratio','pred','pred_Ratio','START_TIME','END_TIME','RUNNING_TIME']
        self.data = self.data[position_columns]

        self.data = RateChangeProcessor.generate_tro_neg_count(self.data)
        self.apply_automation_labeling()
        self.data.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
                             'TRO_DIFF','TRO_PRED','PRED_DIFF','START_TIME','END_TIME','RUNNING_TIME','TRO_NEG_COUNT',
                            'STEEP_LABEL','SLOWLY_LABEL'] 
        
        self.update_tro_condition()
        self.give_tro_out_of_water_condition()
        self.data = StatHungting.label_hunting_multiple_of_two(self.data)
        self.data, self.count = apply_time_offset.classify_time_offset_label(self.data) # 수정


    def apply_fault_label_statistics(self):
        """
        그룹화된 데이터의 통계를 계산하는 추상 메서드
        하위 클래스에서 통계 로직을 구현해야 함
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg(
                                                    {
                                                    'STEEP_LABEL':'sum','SLOWLY_LABEL':'sum','OUT_OF_WATER_STEEP':'sum','HUNTING':'sum','TIME_OFFSET':'sum'
                                                    }
                                                                              ).reset_index()
        self.group['START_TIME'] = self.start_date
        self.group['END_TIME'] = self.end_date
        self.group['RUNNING_TIME'] = self.running_time 
        self.group['OP_TYPE'] = self.op_type 

        self.group = self.group[
                                [
                                'SHIP_ID','OP_INDEX','SECTION','RUNNING_TIME','OP_TYPE','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET','START_TIME','END_TIME']
                                ]
        load_database('signlab','tc_ai_fault_group', self.group)

    def apply_tro_fault_detector(self):
        """ TRO 알고리즘 적용
        Returns: 오퍼레이션 실시간 데이터 자동적재, 오퍼레이션 그룹 자동적재
        """
        self.refine_frames()
        self.predict_tro_val()
        self.apply_tro_labeling()
        self.data = self.data[
            ['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET',
             'START_TIME','END_TIME','RUNNING_TIME','OP_TYPE']
            ]
        
        self.apply_fault_label_statistics()
        self.data = self.data[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET']]
        return self.data, self.group

  
 

