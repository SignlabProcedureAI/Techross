# basic
import pandas as pd
import numpy as np

# type hinting
from typing import Tuple, Callable, Union, Optional

# time
from datetime import datetime

# module.dataline
from prep_dataline.select_dataset import get_dataframe_from_database
from prep_dataline.load_database import load_database

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.indicator_data = data.copy()
        self.indicator_data['duplication'] = False
        self.indicator_data['missing_value'] = False
        self.indicator_data['operation_time'] = False
        self.outlier_ballast_dict = { 
            'CSU':[0,49.46],'STS':[0,33.29],'FTS':[0,39.24],'FMU':[286,2933],'TRO':[0,8],'CURRENT':[0,18790],'VOLTAGE':[3.0,4.7]
            }
        self.outlier_deballast_dict = {
           'CSU':[0,49.46],'STS':[0,33.29],'FMU':[286,2933],'TRO':[0,1.79], 'ANU':[0,1320]
            }
        
    def filter_columns_by_optype(self) -> 'DataPreprocessor' :
        self.op_type = self.data['OP_TYPE'].iloc[0]
        if self.op_type == 2: # Deballast
            self.data = self.data[
                [
                    'SHIP_ID', 'OP_INDEX', 'SECTION', 'OP_TYPE', 
                    'DATA_TIME', 'DATA_INDEX', 'CSU', 'STS', 'FMU', 'TRO', 'ANU'
                ]
            ]
        else: # Ballst
            self.data = self.data[
                [
                    'SHIP_ID', 'OP_INDEX', 'SECTION', 'OP_TYPE', 
                    'DATA_TIME', 'DATA_INDEX', 'CSU', 'STS', 'FMU', 
                    'FTS', 'TRO', 'CURRENT', 'VOLTAGE', 'RATE'
                ]
            ]   
        return self
    
    def remove_duplicates(self) -> Optional['DataPreprocessor']:
        if self.data is None or self.data.empty: 
            return None
        columns_to_check = ['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME']
        duplicates = self.data.duplicated(subset=columns_to_check, keep='first')
        self.indicator_data.loc[self.data[duplicates].index, 'duplication'] = True
        self.data = self.data[~duplicates]

        duplicated_len = len(self.data)
        self.dulication_len = (self.og_sensor_len - duplicated_len)
        ratio=np.round((duplicated_len/self.og_sensor_len)*100,2)
        self.duplication_text = f'전체 데이터의 {np.round(100-ratio,2)}%가 중복,  중복 데이터를 제거함'

        return self 
    
    def remove_missing_values(self) -> Optional['DataPreprocessor']:
        if self.data is None or self.data.empty:
            return None
        source_len = len(self.data)
        self.data = self.data.sort_values(['SHIP_ID','OP_INDEX','SECTION','DATA_TIME'])
        self.data.replace('\\N', pd.NA, inplace=True)
        self.data.dropna(inplace=True)
        missing_indices = self.indicator_data[self.data.isna().any(axis=1)].index
        self.indicator_data.loc[missing_indices, 'missing_value'] = True

        self.missing_len = (source_len - len(self.data))
        ratio = np.round((len(self.data)/source_len)*100,2)
        self.missing_values_text = f'전체 데이터의 {np.round(100-ratio,2)}%가 결측치, 결측치 데이터를 제거함'
        
        return self
    
    def remove_negative_values(self, col: str) -> 'DataPreprocessor':
        self.data = self.data[self.data[col] >= 0]
        return self
    
    def remove_short_operating_time(self, threshold: int = 60) -> Optional['DataPreprocessor']:
        if self.data is None or self.data.empty:
            return None
        source_len = len(self.data)
        self.og_drop_idx_len = len(self.data[self.data['DATA_INDEX']<=60].index) 

        self.data['DATA_INDEX'] = self.data['DATA_INDEX'].astype(int)
        drop_indices = self.indicator_data[self.indicator_data['DATA_INDEX'] <= threshold].index
        self.indicator_data.loc[drop_indices, 'operation_time'] = True
        self.data = self.data[self.data['DATA_INDEX'] > threshold]

        ratio = np.round((self.og_drop_idx_len/source_len)*100,2)
        self.operation_values_text = f'전체 데이터의 {ratio}%에 해당하는 시작 후 60분 데이터를 제거'
        return self
    
    def remove_outliers(self) -> Union[
        Tuple[None, pd.DataFrame],
        'DataPreprocessor'
        ]:
        self.text_dict = {}
        self.noise_sum = 0
    
        if self.op_type != 2: # Ballast
            cols = ['CSU','STS','FTS','FMU','TRO','CURRENT','VOLTAGE']
            outlier_dict = self.outlier_ballast_dict
        else: # Deballast
            cols = ['CSU','STS','FMU','TRO','ANU']
            outlier_dict = self.outlier_deballast_dict

        for col in cols:
            self.data, removed_len  = self.remove_quantile_outliers(col, outlier_dict)
            self.text_dict[col] = self.outlier_values_text
            self.noise_sum += removed_len
            # 아웃라이어 제거 후 데이터가 None이면 함수 종료
            if self.data is None:
                return None, self.indicator_data  
            
        return self
    
    def remove_quantile_outliers(self, col: str, outlier_dict: dict) -> Union[
        None,
        None,
        Tuple[pd.DataFrame, int]
        ]:
        
        if self.data is None or self.data.empty:
            return None

        source_len = len(self.data)

        zero_count = (self.data[col]==0).sum()
        total_count = len(self.data[col])
        zero_ratio = (zero_count/total_count) # 0의 비율이 90% 이상인지 확인

        if zero_ratio >= 0.9:
            self.removed_data = 0
            self.indicator_data[f'{col}_outlier'] = False
            self.outlier_values_text = f'{col} 센서 : 90% 이상 0.'

        thresholds = outlier_dict.get(col)
        bottom_95, top_95 = thresholds

        self.data = self.data[(self.data[col] > bottom_95) & (self.data[col] < top_95)]  # Filter data based on thresholds
        removed_indices = self.data[(self.data[col] <= bottom_95) | (self.data[col] >= top_95)].index
        self.indicator_data[f'{col}_outlier'] = False
        self.indicator_data.loc[removed_indices, f'{col}_outlier'] = True

        # Calculate statistics
        removed_count = source_len - len(self.data)
        ratio_removed = np.round((removed_count / len(self.data)) * 100, 2)
        if len(removed_indices) > 0:
            self.outlier_values_text = f'{col} 센서 제거량: {np.round(100-ratio_removed,2)}%,  범위: {bottom_95} ~ {top_95} 미만 또는 초과,  이상치 데이터 제거'
        else:
            self.outlier_values_text = None
        
        if self.data.empty:
            print("이상 값 제거 후 데이터가 비어 있습니다.")
            return None
        
        return self.data, removed_count

    def organize_data(self) -> 'DataPreprocessor':
        self.data = self.data.sort_values(by='DATA_INDEX').reset_index(drop=True)
        self.og_sensor_len = len(self.data)
        return self
    
    def filter_columns_op_type_final(self) -> 'DataPreprocessor':
        common_columns = [
            'SHIP_ID', 'OP_INDEX', 'SECTION', 'OP_TYPE', 
            'DATA_TIME', 'DATA_INDEX', 'duplication', 
            'missing_value', 'operation_time'
        ]

        if self.op_type != 2:
            specific_columns = [
                'CSU_outlier', 'STS_outlier', 'FTS_outlier', 
                'FMU_outlier', 'TRO_outlier', 'CURRENT_outlier', 'VOLTAGE_outlier'
            ]
            self.indicator_col_data = self.indicator_data[common_columns + specific_columns]
        else:
            specific_columns = [
                'CSU_outlier', 'STS_outlier', 'FMU_outlier', 
                'TRO_outlier', 'ANU_outlier'
            ]
            self.indicator_col_data = self.indicator_data[common_columns + specific_columns]

        return self

    def get_results(self) -> dict:
        if self.data is None or self.data.empty:
            return {'data': None}
        return {
                'data' : self.data, 
                'indicator_data': self.indicator_data, 
                'duplication_text': self.duplication_text, 
                'missing_values_text': self.missing_values_text, 
                'operation_values_text': self.operation_values_text, 
                'text_dict': self.text_dict,
                'duplication_len' : self.dulication_len,
                'missing_len' : self.missing_len,
                'operation_len' :self.og_drop_idx_len,
                'noise_len' : self.noise_sum
        }
    

class PreprocessingPipeline:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.data_len = len(data)

    def apply_pipeline(self) -> dict:
        preprocessor = DataPreprocessor(self.data)
        preprocessor.filter_columns_by_optype() \
                    .organize_data() \
                    .remove_duplicates() \
                    .remove_missing_values() \
                    .remove_short_operating_time() \
                    .remove_outliers() \
                    .filter_columns_op_type_final() \
                    .organize_data()
        results = preprocessor.get_results()

        summary = PreprocessingSummary(results, self.data_len)
        summary.prepare_indicator_summary() \
                .generate_preprocessing_count_dataframe() \
                .create_supplementary_text() 
        summary_results = summary.get_results() 
        summary_results['original_data'] = self.data
        summary_results['processed_data'] = results['data']

        return summary_results


    
class PreprocessingSummary:
    def __init__(self, results: dict, data_len: int) -> None:
        self.data = results.get('data')
        self.indicator_data = results.get('indicator_data')
        self.duplication_text = results.get('duplication_text')
        self.missing_values_text = results.get('missing_values_text')
        self.operation_values_text = results.get('operation_values_text')
        self.text_dict = results.get('text_dict')

        self.data_count = data_len
        self.pre_count = len(self.data)
        self.duplication_len = results.get('duplication_len')
        self.missing_len = results.get('missing_len')
        self.operation_len = results.get('operation_len')
        self.noise_len = results.get('noise_len')

    def prepare_indicator_summary(self) -> 'DataPreprocessor':
        """
        전처리된 데이터 리스트를 처리하여 딕셔너리를 반환하는 함수.
        """
        # 데이터 복사 및 시간 형식 변환
        # self.indicator_data = self.indicator_data.copy()
        self.indicator_data['DATA_TIME'] = pd.to_datetime(self.indicator_data['DATA_TIME'])

        # 딕셔너리 초기화 
        self.indicator_dict = {
            'SHIP_ID': None, 'OP_INDEX': None, 'SECTION': None, 
            'OP_TYPE': None, 'DATA_COUNT': None, 'PRE_COUNT': None, 
            'START_DATE': None, 'END_DATE': None, 'REG_DATE': None
        }

        # 고정 값 설정
        first_row = self.indicator_data.iloc[0]
        self.indicator_dict.update({
            'SHIP_ID': first_row['SHIP_ID'],
            'OP_INDEX': first_row['OP_INDEX'],
            'SECTION': first_row['SECTION'],
            'OP_TYPE': first_row['OP_TYPE'],
        })

        # 날짜 값 설정
        self.indicator_dict['START_DATE'] = first_row['DATA_TIME']
        self.indicator_dict['END_DATE'] = self.indicator_data.iloc[-1]['DATA_TIME']
        self.indicator_dict['REG_DATE'] = datetime.now()

        # 데이터 개수 설정
        self.indicator_dict['DATA_COUNT'] = self.data_count
        self.indicator_dict['PRE_COUNT'] = self.pre_count

        return self

    def generate_preprocessing_count_dataframe(self) -> 'DataPreprocessor':
        count_dict = [{'NOISE': self.noise_len ,'MISSING': self.missing_len, 'DUPLICATE': self.duplication_len, 'OPERATION': self.operation_len}]
        self.count_df = pd.DataFrame(count_dict)

        return self

    def create_supplementary_text(self) -> 'DataPreprocessor':
        """
        각 전처리 지표에 대한 보조 설명 텍스트를 생성하여 딕셔너리로 반환합니다.
        """
        # 텍스트 매핑
        conditions = {
            'noise': (self.count_df['NOISE'].iloc[0] > 0, self.text_dict),
            'missing': (self.count_df['MISSING'].iloc[0] > 0, self.missing_values_text),
            'duplication': (self.count_df['DUPLICATE'].iloc[0] > 0, self.duplication_text),
            'operation': (self.count_df['OPERATION'].iloc[0] > 0, self.operation_values_text),
        }

        # 조건에 따라 텍스트 생성
        self.text_dict = {key: value if condition else None for key, (condition, value) in conditions.items()}

        return self

    def get_results(self):
        return {
            'indicator_dict': self.indicator_dict, 
            'count_df' : self.count_df,
            'text_dict' : self.text_dict
        }


