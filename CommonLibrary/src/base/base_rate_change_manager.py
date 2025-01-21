from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
from typing import Any, Optional

class DataUtility(ABC):
    """
    데이터 처리를 위한 기본 클래스
    """
    @abstractmethod
    def calculate_rate_change(self, df:pd.DataFrame, column: str) -> pd.DataFrame:
        """비율 변화를 계산하는 추상 메서드
        """
        pass

    @staticmethod
    def generate_rolling_mean(data: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
        data.sort_index(inplace=True)
        data['rolling_mean'] = data[col].rolling(window=window).mean()
        data.dropna(inplace=True)
        return data 
    
    @staticmethod
    def load_pickle(file_path: str) -> Optional[Any]:
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            return data
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None
    
    @staticmethod
    def generate_tro_neg_count(abnormal: pd.DataFrame) -> pd.DataFrame:
        abnormal['TRO_NEG_COUNT'] = 0
        negative_count = 0

        def calculate_true_difference(tro_value: float, predicted_value: float) -> float:
            return np.abs(tro_value - predicted_value)

        def is_decline(tro_value: float, predicted_value: float, temporary_increase_threshold: float) -> bool:
            true_diff = calculate_true_difference(tro_value, predicted_value)
            return (tro_value < temporary_increase_threshold) and (tro_value < predicted_value) and (true_diff > 0.24)

        def has_temporary_increase(tro_value: float) -> bool:
            temporary_increase_threshold = 0.13
            return tro_value > temporary_increase_threshold

        def negative_has_temporary_increase(tro_value: float, predicted_value: float) -> bool:
            temporary_increase_threshold = -0.13
            return is_decline(tro_value, predicted_value, temporary_increase_threshold)

        for i in range(len(abnormal)):
            tro_value = abnormal.iloc[i, abnormal.columns.get_loc('TRO_Ratio')]
            predicted_value = abnormal.iloc[i, abnormal.columns.get_loc('pred_Ratio')]

            temporary_increase_check = negative_has_temporary_increase(tro_value, predicted_value)
            if tro_value < -1.5 and temporary_increase_check:
                negative_count += 1
            else:
                temporary_increase = has_temporary_increase(tro_value)
                if tro_value > 0 and temporary_increase:
                    negative_count = 0

            abnormal.iloc[i, abnormal.columns.get_loc('TRO_NEG_COUNT')] = negative_count

        return abnormal
    
    @staticmethod
    def calculate_true_difference(delta: float, predicted_delta: float) -> float:
        return np.abs(delta - predicted_delta)
    
    @staticmethod
    def is_steep_decline(delta: float, predicted_delta: float) -> bool:
        true_diff = DataUtility.calculate_true_difference(delta, predicted_delta)
        return delta <= -5 and (delta < predicted_delta) and (true_diff > 0.24)

    @staticmethod
    def classify_decline_steep_label(delta: float, predicted_delta: float) -> int:
        return 1 if DataUtility.is_steep_decline(delta, predicted_delta) else 0

    @staticmethod
    def is_slowly_decline(pre_neg_count: int, neg_count: int) -> bool:
        return (neg_count % 3 == 0) and (neg_count != 0) and (pre_neg_count != neg_count)

    @staticmethod
    def classify_decline_slowly_label(pre_neg_count: int, neg_count: int) -> int:
        return 1 if DataUtility.is_slowly_decline(pre_neg_count, neg_count) else 0

    @staticmethod
    def give_tro_condition(data: pd.DataFrame) -> pd.DataFrame:
        data.loc[data['TRO'] >= 8, 'STEEP_LABEL'] = 0
        return data

    @staticmethod
    def give_tro_out_of_water_condition(data: pd.DataFrame) -> pd.DataFrame:
        data['OUT_OF_WATER_STEEP'] = 0
        data.loc[(data['TRO'] <= 1) & (data['STEEP_LABEL'] == 1), 'OUT_OF_WATER_STEEP'] = 1
        data.loc[(data['TRO'] <= 1) & (data['STEEP_LABEL'] == 1), 'STEEP_LABEL'] = 0
        return data
