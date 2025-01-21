
from base import BaseTimeOffset
import numpy as np
from typing import  Tuple
import pandas as pd

class TimeOffsetWithAutocorr(BaseTimeOffset):
    def _format_return(self, autocorr: np.ndarray, count: int) -> Tuple[pd.DataFrame, int]:
        """
        데이터프레임과 자기상관 배열을 반환합니다.

        :param autocorr: 자기상관 배열
        :return: 데이터프레임과 자기상관 배열
        """
        self.data['RE_CROSS_CORRELATION'] = autocorr
        return self.data, count