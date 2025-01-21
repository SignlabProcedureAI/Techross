# basic
import pandas as pd
import pickle

# type hiting
from typing import Tuple, Union
from sklearn.base import BaseEstimator

# module
from base import BaseFaultAlgorithm
from stat_dataline import load_database
from .rate_change_manager import RateChangeProcessor
from .hunting_processor import StatHungting
from .timeoffset_processor import TimeOffsetSimple

class TROFaultAlgorithm(BaseFaultAlgorithm):
    def apply_tro_labeling(self) -> None:
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
        hunting_instance = StatHungting()
        self.data = hunting_instance.label_hunting_multiple_of_two(self.data)
        time_offset_processor = TimeOffsetSimple(self.data)
        self.data = time_offset_processor.classify_time_offset_label() 

    def apply_fault_label_statistics(self) -> None:
        """
        그룹화된 데이터의 통계를 계산하여 저장하는 함수.

        Steps:
            1. 데이터를 그룹화하여 라벨링 통계 계산.
            2. 추가 메타데이터(시작 시간, 종료 시간, 실행 시간, 작업 유형) 추가.
            3. 필요한 열만 선택하여 데이터 정리.
            4. 결과를 데이터베이스에 저장.
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
        # load_database('signlab','tc_ai_fault_group', self.group)
        load_database.load_database('ecs_test','test_tc_ai_fault_group', self.group)

    def apply_tro_fault_detector(self, status: bool) -> Union[None, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        TRO 알고리즘을 적용하여 실시간 데이터와 그룹 데이터를 반환하는 함수.

        Steps:
            1. 데이터 프레임 정제.
            2. TRO 값 예측 및 라벨링.
            3. 실시간 데이터의 필요한 열 선택.
            4. 그룹화된 통계 계산.
            5. 그룹 데이터 정리 후 반환.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - 실시간 데이터 (`self.data`): 정제된 실시간 데이터.
                - 그룹 데이터 (`self.group`): 그룹화된 통계 데이터.
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

        if not status:
            return self.data, self.group

  
    @staticmethod
    def load_model_from_pickle(file_path: str) -> BaseEstimator:
        """
        피클 파일에서 모델을 불러오는 함수.

        Args:
        - file_path: 불러올 피클 파일의 경로

        Returns:
        - model: 불러온 모델 객체
        """
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"모델이 {file_path}에서 성공적으로 불러와졌습니다.")
        return model


