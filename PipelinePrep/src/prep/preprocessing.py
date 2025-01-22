# basic
import pandas as pd
import numpy as np

# type hinting
from typing import Self, Tuple, Callable, Union, Optional

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
        
    def filter_columns_by_optype(self) -> Self:
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
    
    def remove_duplicates(self) -> Optional[Self]:
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
    
    def remove_missing_values(self) -> Optional[Self]:
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
    
    def remove_negative_values(self, col: str) -> Self:
        self.data = self.data[self.data[col] >= 0]
        return self
    
    def remove_short_operating_time(self, threshold: int = 60) -> Optional[Self]:
        if self.data is None or self.data.empty:
            return None
        source_len = len(self.data)
        self.og_drop_idx_len = len(self.data[self.data['DATA_INDEX']<=60].index)

        self.data['DATA_INDEX'] = self.data['DATA_INDEX'].astype(int)
        drop_indices = self.indicator_data[self.indicator_data['DATA_INDEX'] <= threshold].index
        self.indicator_data.loc[drop_indices, 'operation_time'] = True
        self.data = self.data[self.data['DATA_INDEX'] > threshold]

        ratio = np.round((len(self.og_drop_idx_len)/source_len)*100,2)
        self.operation_values_text = f'전체 데이터의 {ratio}%에 해당하는 시작 후 60분 데이터를 제거'
        return self
    
    def remove_outliers(self) -> Union[
        Tuple[None, pd.DataFrame],
        Self
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
            self.text_dict[col] = self.operation_values_text
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
            outlier_values_text = f'{col} 센서 : 90% 이상 0.'

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

    def organize_data(self) -> Self:
        self.data = self.data.sort_values(by='DATA_INDEX').reset_index(drop=True)
        self.og_sensor_len = len(self.data)
        return self
    
    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.data, self.indicator_data
    

class PreprocessingPipeline:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def apply_pipeline(self) -> Callable[[],Tuple[pd.DataFrame, pd.DataFrame] ]:
        preprocessor = DataPreprocessor(self.data)
        preprocessor.filter_columns_by_optype() \
                    .organize_data() \
                    .remove_duplicates() \
                    .remove_missing_values() \
                    .remove_short_operating_time() 
        return preprocessor.get_results()
    