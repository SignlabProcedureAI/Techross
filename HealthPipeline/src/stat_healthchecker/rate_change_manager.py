import numpy as np
import pandas as pd
from CommonLibrary  import DataUtility

class RateChangeProcessor(DataUtility):
     def calculate_rate_change(self, df:pd.DataFrame, column: str) -> pd.DataFrame:
        """비율 변화를 계산하는 추상 메서드
        """
        # 이전 값과의 차이를 계산
        differences = df[column] - df[column].shift(1)

        # 계산된 비율을 새 열로 추가
        df['{}_Ratio'.format(column)] = differences

        return df
