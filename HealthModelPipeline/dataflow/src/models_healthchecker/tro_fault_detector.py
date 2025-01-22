# basic
import os
import pandas as pd
import pickle

# type hinting 
from typing import Union, Tuple
from sklearn.base import BaseEstimator

# module
from base import BaseFaultAlgorithm
from models_dataline import load_database
from .rate_change_manager import RateChangeProcessor
from .hunting_processor import ModelHunting
from .timeoffset_processor import TimeOffsetWithAutocorr

class TROFaultAlgorithm(BaseFaultAlgorithm):
    def catorize_health_score(self) -> None:
        """
        건강 점수를 기반으로 결함 위험 카테고리를 분류하는 함수

        Description:
            - HEALTH_SCORE 값에 따라 각 데이터 포인트를 'NORMAL', 'WARNING', 'RISK', 'DEFECT' 카테고리로 분류합니다
            - 분류 결과는 'RISK' 열에 저장됩니다.
        """
        self.group['SUM'] = self.group['STEEP_LABEL'] + self.group['SLOWLY_LABEL'] + self.group['OUT_OF_WATER_STEEP'] + self.group['HUNTING'] + self.group['TIME_OFFSET']
        
        self.group['DEFECT_RISK_CATEGORY'] = 0

        self.group.loc[self.group['SUM']<=0, 'RISK'] = 'NORMAL'
        self.group.loc[(self.group['SUM']>0) & (self.group['SUM']<=2), 'RISK'] = 'WARNING'
        self.group.loc[(self.group['SUM']>2) & (self.group['SUM']<=6), 'RISK'] = 'RISK'
        self.group.loc[self.group['SUM']>6, 'RISK'] = 'DEFECT'

        self.group = self.group.drop(columns='SUM')

    def predict_stats_val(self) -> None:
        """
        TRO 그룹 모델을 사용하여 통계 데이터를 기반으로 예측 값을 계산하는 함수.

        Args:
            없음

        Returns:
            None: 예측 결과를 그룹 데이터프레임(self.group)에 추가하고,
                최종적으로 필요한 열만 포함하는 데이터프레임으로 업데이트.

        Notes:
            - 모델은 사전에 저장된 피클 파일에서 로드됩니다.
            - 예측 결과는 'PRED' 열로 저장됩니다.
        """
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
        """
        TRO 데이터를 라벨링하고 추가 계산 및 조건을 적용하는 함수.

        Steps:
            1. TRO 및 PRED 열의 변화율 계산.
            2. 필요한 열만 선택하여 데이터 정리.
            3. TRO 음수 개수 계산 및 자동 라벨링 적용.
            4. TRO 조건 업데이트 및 추가 조건 설정.
            5. 헌팅 라벨 및 시간 오프셋 라벨링 적용.
        """
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'TRO')
        self.data = RateChangeProcessor.calculate_rate_change(self.data, 'pred')

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
        model_hunting_instance = ModelHunting()
        self.data = model_hunting_instance.label_hunting_multiple_of_two(self.data)
        time_offset_processor = TimeOffsetWithAutocorr(self.data)
        self.data, self.count = time_offset_processor.classify_time_offset_label() 


    def apply_fault_label_statistics(self):
        """
        그룹화된 데이터의 통계를 계산하고 라벨링 결과를 처리하는 함수.

        Steps:
            1. 데이터 그룹화 및 통계 계산.
            2. 다중 인덱스 열을 단일 레벨로 평탄화.
            3. 추가 메타데이터(시간, 실행 시간, 작업 유형) 추가.
            4. 최종 데이터 저장 및 예측, 라벨링 적용.

        Returns:
            None: self.group가 최종 결과로 업데이트되고 데이터베이스에 저장됩니다.
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
        self.group.columns = ['SHIP_ID','OP_INDEX','SECTION',
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
        load_database('ecs_test','test_tc_ai_fault_group_v1.1.0', '200', self.group)

        self.predict_stats_val()
        self.catorize_health_score()
        # load_database('signlab','tc_ai_fault_model_group', 'release', self.group)
        load_database('ecs_test','test_tc_ai_fault_model_group', '200', self.group)

    def apply_tro_fault_detector(self, status) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        TRO 알고리즘을 적용하여 데이터 정제, 예측, 라벨링 및 통계 계산을 수행하는 함수.

        Args:
            status (bool): 상태 플래그. 특정 동작을 제어할 때 사용.

        Steps:
            1. 데이터 프레임 정제.
            2. TRO 값 예측 및 라벨링 적용.
            3. 필요한 열 선택으로 데이터 정리.
            4. 그룹화된 통계 계산.
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

        if not status:
            return self.data, self.group
  


