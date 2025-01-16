from CommonLibrary import BaseTimeOffset
import numpy as np
from typing import  Tuple
import pandas as pd

class TimeOffsetSimple(BaseTimeOffset):
    def _format_return(self, autocorr: np.ndarray) -> pd.DataFrame:
        """
        처리된 데이터프레임만 반환합니다.

        :param autocorr: 자기상관 배열
        :return: 데이터프레임
        """
        return self.data
