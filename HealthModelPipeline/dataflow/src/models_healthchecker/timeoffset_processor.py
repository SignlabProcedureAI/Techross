
from CommonLibrary import BaseTimeOffset
import numpy as np
from typing import  Tuple
import pandas as pd

class TimeOffsetWithAutocorr(BaseTimeOffset):
    def _format_return(self, autocorr: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        데이터프레임과 자기상관 배열을 반환합니다.

        :param autocorr: 자기상관 배열
        :return: 데이터프레임과 자기상관 배열
        """
        self.data['RE_CROSS_CORRELATION'] = autocorr
        return self.data, autocorr