import numpy as np
from scipy.signal import find_peaks, correlate
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Union
from abc import ABC, abstractmethod
from scipy.fft import fft, fftfreq

class BaseTimeOffset(ABC):
     """ timeoffset 공통 기능을 담당하는 베이스 클래스
     """
     def __init__(self, data: pd.DataFrame) -> None:
          """
          데이터 초기화 및 공통 속성 설정

          Args: 입력 데이터프레임
          """
          self.data = data
   
     def classify_time_offset_label(self) -> Union[Tuple[pd.DataFrame, int], pd.DataFrame]:
          """
          데이터에서 시간 오프셋 레이블을 분류

          Returns: 
          처리된 데이터프레임과 선택적으로 자기상관 배열
          """
          tro = self.data['TRO'].to_numpy()
          current = self.data['CURRENT'].to_numpy()

          # 신호 정규화
          current_normalized = (current - np.mean(current)) / np.std(current)
          tro_normalized = (tro - np.mean(tro)) / np.std(tro)

          # 크로스 코릴레이션 계산
          cross_correlation = correlate(tro_normalized, current_normalized, mode='full')
          lags_full = np.arange(-(len(current_normalized) - 1), len(tro_normalized))

          # 양의 lags 추출
          positive_lags = lags_full[lags_full >= 0]
          positive_cross_correlation = cross_correlation[lags_full >= 0]
          
          self.data['CROSS_CORRELATION'] = positive_cross_correlation[:len(current_normalized)]
          self.data['LAG'] = positive_lags[:len(current_normalized)]

          # 임계값 적용
          self.apply_threshold()

          # 자기상관 계산
          corrlat = self.data['CROSS_CORRELATION'].values
          autocorr = correlate(corrlat, corrlat, mode='full')[len(corrlat) - 1:]

          # 반복 극값 필터링
          result, count = self.filter_repeating_extrema(autocorr)
          self.evaluate_time_offset(result)

          # 초기 시간 필터 제한
          self.limit_date_time()

          return self._format_return(autocorr, count)
   
     def apply_threshold(self) -> None:
          """
          크로스 코릴레이션 값에 임계값을 적용합니다.
          """
          cross_correlation = self.data['CROSS_CORRELATION']
          peaks, _ = find_peaks(cross_correlation, height=30, prominence=30)

          self.data['TIME_OFFSET'] = 0
          for i in range(2, len(peaks), 3):
               self.data.loc[peaks[i], 'TIME_OFFSET'] = 1
   
     def filter_repeating_extrema(self, data: np.ndarray, repeat_threshold: int = 4) -> Tuple[bool, int]:
          """
          반복되는 극값 구간을 필터링합니다.

          Args:
           - data: 자기상관 데이터 배열
           - repeat_threshold: 반복 허용 임계값
          Returns: 
           극값이 반복되는지 여부
          """
          lower_value = np.percentile(data, 20)
          upper_value = np.percentile(data, 80)

          maxima, _ = find_peaks(data, height=upper_value, prominence=upper_value / 4)
          minima, _ = find_peaks(-data, height=-lower_value, prominence=-lower_value / 4)

          extrema_indices = np.sort(np.concatenate([maxima, minima]))

          count = 0
          alternating = False
          for i in range(1, len(extrema_indices)):
               if (extrema_indices[i] in maxima and extrema_indices[i - 1] in minima) or \
                    (extrema_indices[i] in minima and extrema_indices[i - 1] in maxima):
                    count += 1
                    alternating = True
               else:
                    if count >= repeat_threshold:
                         return True, count
                    count = 1
                    alternating = False

          if alternating and count >= repeat_threshold:
               return True, count

          return False, count

     def limit_date_time(self) -> None:
          """
          초기 시간 구간의 데이터를 제한합니다.
          """
          self.data.loc[self.data['DATA_INDEX'] < 30, 'TIME_OFFSET'] = 0

     
     def evaluate_time_offset(self, result: bool) -> None:
        """
        시간 오프셋 평가 결과를 적용합니다.

        Args: 
         필터링 결과 (True/False)
        """
        if not result:
            self.data['TIME_OFFSET'] = 0

     @abstractmethod
     def _format_return(self, autocorr: np.ndarray, count: int):
          """
          반환값 형식을 정의합니다 (자식 클래스에서 구현 필요).

          Args:
           - autocorr: 자기상관 배열
          Returns: 
           - 정의된 반환값
          """
          pass


class TimeOffsetVisualizer:
     @staticmethod
     def plot_cross_correlation(data: pd.DataFrame) -> None:
          """
          크로스 코릴레이션을 시각화합니다.

          Args:
           data: 입력 데이터프레임
          """
          cross_correlation = data['cross_correlation']
          lags = data['lag']
          plt.plot(lags, cross_correlation)
          plt.title('Cross-correlation')
          plt.xlabel('Lag')
          plt.ylabel('Correlation coefficient')
          plt.show()

     @staticmethod
     def plot_peak_cross_correlation(data: pd.DataFrame) -> None:
          """
          크로스 코릴레이션과 피크를 시각화합니다.

          Args:
           data: 입력 데이터프레임
          """
          cross_correlation = data['cross_correlation']
          peaks, _ = find_peaks(cross_correlation, height=30, prominence=30)
          plt.plot(cross_correlation, label='Cross-correlation')
          plt.scatter(peaks, cross_correlation[peaks], color='red', label='Detected Peaks')
          plt.title('Cross-correlation with Detected Peaks')
          plt.xlabel('Lag')
          plt.ylabel('Cross-correlation Coefficient')
          plt.legend()
          plt.show()

     @staticmethod
     def plot_peak_cross_correlation(data: pd.DataFrame) -> None:   

          cross_correlation=data['cross_correlation']
          peaks, _ = find_peaks(cross_correlation, height=30,prominence=30)

          plt.figure(figsize=(14, 7))
          plt.plot(cross_correlation, label='Cross-correlation')
          plt.scatter(peaks, cross_correlation[peaks], color='red', label='Detected Peaks')
          plt.title('Cross-correlation with Detected Peaks')
          plt.xlabel('Lag')
          plt.ylabel('Cross-correlation Coefficient')
          plt.legend()
          plt.show()

     @staticmethod
     def plot_sample_corr_fft(data: np.ndarray) -> None:
          """
          원본 데이터, FFT 파워 스펙트럼, 자기상관 결과를 시각화합니다.

          Args: 
           입력 데이터 배열
          """
          # FFT Analysis
          N = len(data)
          T = 1.0 / N
          yf = fft(data)
          xf = fftfreq(N, T)[:N // 2]

          # Power spectrum
          power = np.abs(yf[:N // 2])

          # Autocorrelation
          autocorr = correlate(data, data, mode='full')
          autocorr = autocorr[autocorr.size // 2:]

          # Plot all together: Original Data, FFT, and Autocorrelation
          plt.figure(figsize=(18, 6))

          # Plot original data
          plt.subplot(1, 3, 1)
          plt.plot(data)
          plt.title('Original Data')
          plt.xlabel('Sample')
          plt.ylabel('Value')

          # Plot FFT Power Spectrum
          plt.subplot(1, 3, 2)
          plt.plot(xf, power)
          plt.title('FFT Power Spectrum')
          plt.xlabel('Frequency')
          plt.ylabel('Power')
          plt.grid(True)

          # Plot Autocorrelation
          plt.subplot(1, 3, 3)
          plt.plot(autocorr)

          lower_value = np.percentile(autocorr, 20)
          upper_value = np.percentile(autocorr, 80)

          print(upper_value)

          peaks, _ = find_peaks(autocorr, height=upper_value, prominence=upper_value / 4, distance=len(autocorr) / 10)
          re_peaks, _ = find_peaks(-autocorr, height=-lower_value, prominence=-lower_value / 4, distance=len(autocorr) / 10)

          print(peaks)
          print(re_peaks)

          plt.plot(peaks, autocorr[peaks], "rx")
          plt.plot(re_peaks, autocorr[re_peaks], "rx")

          plt.title('Autocorrelation with Detected Peaks')
          plt.xlabel('Lag')
          plt.ylabel('Autocorrelation')

          plt.tight_layout()
          plt.show()
