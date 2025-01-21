import numpy as np
import pandas as pd
from base import BaseHunting
from .rate_change_manager import RateChangeProcessor

class ModelHunting(BaseHunting):
    @staticmethod
    def calculate_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
        """
        ModelHunting 전용 변화율 계산
        """
        return RateChangeProcessor.calculating_rate_change(df, 'TRO')
    
    @staticmethod
    def add_custom_logic(df: pd.DataFrame) -> pd.DataFrame:
        """
        추가 로직: PEAK_VALLEY_INDICES 생성
        """
        peaks_indices = df[df['Peak']].index
        valleys_indices = df[df['Valley']].index
        df['PEAK_VALLEY_INDICES'] = 0
        df.loc[np.sort(np.concatenate((peaks_indices, valleys_indices))), 'PEAK_VALLEY_INDICES'] = 1
        return df