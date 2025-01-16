from CommonLibrary import BaseFaultAlgorithm
import pandas as pd
from  models_dataline import load_database
import os
from rate_change_manager import RateChangeProcessor
from hunting_processor import ModelHunting
from timeoffset_processor import TimeOffsetWithAutocorr
from typing import Union, Tuple

class TROFaultAlgorithm(BaseFaultAlgorithm):
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data (pd.DataFrame): 처리할 데이터프레임
        """
        super().__init__(data)

    def catorize_health_score(self) -> None:

        self.group['SUM'] = self.group['STEEP_LABEL'] + self.group['SLOWLY_LABEL'] + self.group['OUT_OF_WATER_STEEP'] + self.group['HUNTING'] + self.group['TIME_OFFSET']
        
        self.group['DEFECT_RISK_CATEGORY'] = 0

        self.group.loc[self.group['SUM']<=0, 'RISK'] = 'NORMAL'
        self.group.loc[(self.group['SUM']>0) & (self.group['SUM']<=2), 'RISK'] = 'WARNING'
        self.group.loc[(self.group['SUM']>2) & (self.group['SUM']<=6), 'RISK'] = 'RISK'
        self.group.loc[self.group['SUM']>6, 'RISK'] = 'DEFECT'

        self.group = self.group.drop(columns='SUM')

    def predict_stats_val(self) -> None:
        tro_model_relative_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models_model/tro_group_model_v2.0.0')
        tro_model_path = os.path.abspath(tro_model_relative_path)
        model = self.load_model_from_pickle(tro_model_path)

        X = self.group[['CSU', 'STS', 'FTS', 'FMU', 'CURRENT', 'TRO_MIN', 'TRO_MEAN', 'TRO_MAX',
       'TRO_DIFF_MIN', 'TRO_DIFF_MEAN', 'TRO_DIFF_MAX', 'TRO_NEG_COUNT',
       'PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION',
       'RE_CROSS_CORRELATION_COUNT']]
        
        self.group['PRED'] =  model.predict(X)

        self.group = self.group[
            ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET',
             'PRED','START_TIME','END_TIME','RUNNING_TIME']
            ]

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
        self.data = ModelHunting.label_hunting_multiple_of_two(self.data)
        time_offset_processor = TimeOffsetWithAutocorr(self.data)
        self.data, self.count = time_offset_processor.classify_time_offset_label() 


    def apply_fault_label_statistics(self):
        """
        그룹화된 데이터의 통계를 계산하는 추상 메서드
        하위 클래스에서 통계 로직을 구현해야 함
        """
        self.group = self.data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg(
                                                    {
                                                                            'CSU':'mean','STS':'mean','FTS':'mean','FMU':'mean','CURRENT':
                                                             'mean','TRO':['min','mean','max'],'TRO_DIFF':['min','mean','max'],'TRO_PRED':'mean','PEAK_VALLEY_INDICES':'sum',
                                                             'CROSS_CORRELATION':'mean','RE_CROSS_CORRELATION':'mean','PRED_DIFF':'mean','TRO_NEG_COUNT':'max',
                                                             'STEEP_LABEL':'sum','SLOWLY_LABEL':'sum','OUT_OF_WATER_STEEP':'sum','HUNTING':'sum','TIME_OFFSET':'sum'
                                                    }
                                                                              ).reset_index()
        
        # 다중 인덱스 컬럼 → 단일 레벨 평탄화
        self.group.columns = ['_'.join(col) for col in self.group.columns]
        self.group.columns = [
                            'CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX','TRO_DIFF_MIN','TRO_DIFF_MEAN',
                            'TRO_DIFF_MAX','TRO_PRED','PEAK_VALLEY_INDICES_SUM','CROSS_CORRELATION','RE_CROSS_CORRELATION','PRED_DIFF',
                            'TRO_NEG_COUNT','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET'
                            ]
        
        self.group['START_TIME'] = self.start_date
        self.group['END_TIME'] = self.end_date
        self.group['RUNNING_TIME'] = self.running_time 
        self.group['OP_TYPE'] = self.op_type 
        self.group['RE_CROSS_CORRELATION_COUNT'] = self.count

        self.group = self.group[
                                ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX',
                                 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','PEAK_VALLEY_INDICES_SUM','CROSS_CORRELATION','RE_CROSS_CORRELATION',
                                 'PRED_DIFF','RE_CROSS_CORRELATION_COUNT','TRO_NEG_COUNT','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING',
                                 'TIME_OFFSET','START_TIME','END_TIME','RUNNING_TIME']
                                ]
        load_database('ecs_test','tc_ai_fault_group_v1.1.0', '200', self.group)

        self.predict_stats_val()
        self.group = self.catorize_health_score(self.group)
        load_database('signlab','tc_ai_fault_model_group', 'release', self.group)

    def apply_tro_fault_detector(self,status) -> None:
        """ TRO 알고리즘 적용
        Returns: 오퍼레이션 실시간 데이터 자동적재, 오퍼레이션 그룹 자동적재
        """
        self.refine_frames()
        self.predict_tro_val()
        self.apply_tro_labeling()
        self.data = self.data[
            ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','TRO_DIFF','TRO_PRED','PRED_DIFF',
            'PEAK_VALLEY_INDICES','RE_CROSS_CORRELATION','CROSS_CORRELATION','TRO_NEG_COUNT','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING',
            'TIME_OFFSET','START_TIME','END_TIME','RUNNING_TIME']
            ]
        self.apply_fault_label_statistics()
  
 

