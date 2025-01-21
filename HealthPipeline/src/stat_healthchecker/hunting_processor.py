import pandas as pd
from base import BaseHunting
from .rate_change_manager import RateChangeProcessor

class StatHungting(BaseHunting):
    @staticmethod
    def calculate_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
        """
        StatHunting 전용 변화율 계산
        """
        return RateChangeProcessor.calculate_rate_change(df, 'TRO')
        