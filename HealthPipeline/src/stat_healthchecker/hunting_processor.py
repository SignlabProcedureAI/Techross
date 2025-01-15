import pandas as pd
from CommonLibrary import BaseHunting
from rate_change_manager import RateChangeProcessor

class StatHungting(BaseHunting):
    @staticmethod
    def calculate_rate_of_change(df: pd.DataFrame) -> pd.DataFrame:
        """
        StatHunting 전용 변화율 계산
        """
        return RateChangeProcessor.calculating_rate_change(df, 'TRO')
        