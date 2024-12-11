# basic
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt


# module.healthchecker
import stat_healthchecker.rate_of_change_algorithms as rate_algorithms

# math
from scipy.signal import correlate
from scipy.signal import find_peaks


def classify_time_offset_label(data):
    # 변수 설정
    tro=data['TRO'].to_numpy()
    current=data['CURRENT'].to_numpy()
    
    # 신호 정규화
    current_normalized = (current - np.mean(current)) / np.std(current)
    tro_normalized = (tro - np.mean(tro)) / np.std(tro)

    # 크로스 코릴레이션 계산current_normalized
    cross_correlation = correlate(tro_normalized, current_normalized, mode='full')
    
    # 최대 코릴레이션 인덱스 찾기
    offset = cross_correlation.argmax() - (len(tro) - 1)
    
    # 'full' 모드의 lags 배열을 생성합니다.
    lags_full = np.arange(-(len(current_normalized) - 1), len(tro_normalized))
    
    # 음수 lags를 제거하여 양수 lags와 그에 해당하는 크로스 코릴레이션 값을 추출합니다.
    positive_lags = lags_full[lags_full >= 0]
    positive_cross_correlation = cross_correlation[lags_full >= 0]

    # 양수 lags에 대한 크로스 코릴레이션을 데이터 프레임에 매핑하기 위해 인덱스를 조정합니다.
    # 주의: 이렇게 조정하면 양수 lags가 실제 데이터의 인덱스와 정확히 일치합니다.
    data['cross_correlation']=positive_cross_correlation[:len(current_normalized)]
    data['lag']=positive_lags[:len(current_normalized)]

    # 임계 값 함수 적용
    data=apply_threshold(data)
    
    # cross_correlation 변수 저장
    corrlat = data['cross_correlation'].values

    # cross_correlation 활용 re_cross_correlation 추출 
    autocorr = correlate(corrlat, corrlat, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    # filter_repeating_extrema 활용 re_cross_correlation의 극대, 극소 주기 카운트
    result = filter_repeating_extrema(autocorr)

    # re_cross_corrlation 평가
    data = evaluate_time_offset(result,data)

    # 데이터 초기 시간 필터링
    data = limit_date_time(data)
    
    return data


def apply_threshold(data):
    
    cross_correlation=data['cross_correlation']
    
    cross_max=cross_correlation.max()
    
    threshold = (cross_max / 2)
    
    peaks, rest = find_peaks(cross_correlation, height=30,  prominence=30)  # height=0은 모든 양의 봉우리를 찾는데 사용됨
    
    # TIME OFFSET 변수를 데이터 프레임에 추가하고 모든 값을 False로 초기화
    data['TIME_OFFSET'] = 0
          
    for i in range(2,len(peaks),3):
        data.loc[peaks[i], 'TIME_OFFSET'] = 1
    
    return data

    
def limit_date_time(data):
    data.loc[data['DATA_INDEX'] < 30,'TIME_OFFSET'] = 0
    
    return data

def filter_repeating_extrema(data, repeat_threshold=4):
    """
    Filters out sections of the data where local minima and maxima repeat more than a specified threshold.

    :param data: The input signal data.(자기상관계수)
    :param repeat_threshold: The minimum number of times extrema must repeat to be filtered.
    :return: Filtered data with repeating extrema sections removed.
    """
    # Find local maxima and minima
    lower_value = np.percentile(data, 20)
    upper_value = np.percentile(data, 80)
    
    maxima, _ = find_peaks(data,  height=upper_value,  prominence=upper_value/4, distance=len(data)/10)
    minima, _ = find_peaks(-data, height=-lower_value,  prominence=(-lower_value/4), distance=len(data)/10)
    
    # print(f'극대 값: {maxima}')
    # print(f'극소 값: {minima}')
    
    # Combine and sort indices of extrema
    extrema_indices = np.sort(np.concatenate([maxima, minima]))
    
    # Check for alternating extrema sequence
    count = 0
    alternating = False

    for i in range(1, len(extrema_indices)):
        # Check if current and previous extrema are alternating
        if (extrema_indices[i] in maxima and extrema_indices[i-1] in minima) or \
           (extrema_indices[i] in minima and extrema_indices[i-1] in maxima):
            count += 1
            alternating = True
        else:
            # Reset count if the alternation breaks
            if count >= repeat_threshold:
                return True
            count = 1
            alternating = False

    # Final check in case the last segment meets the criteria
    if alternating and count >= repeat_threshold:
        return True

    return False

def evaluate_time_offset(result,data):
    
    if result:
        return data
    else:
        data['TIME_OFFSET'] = 0
        return data
    

### 시각화 함수

def plot_cross_correlation(data):
    
    #변수 설정
    cross_correlation=data['cross_correlation']
    
    # 최대 코릴레이션 인덱스 찾기
    #offset = cross_correlation.argmax() - (len(data) - 1)

    # 결과 플롯
    lags = data['lag']
    plt.plot(lags[lags >= 0], cross_correlation[lags >= 0])
    plt.title('Cross-correlation')
    plt.xlabel('Lag')
    plt.ylabel('Correlation coefficient')
    plt.show()

    #print(f"The offset is: {offset} time units")

    
def plot_peak_cross_correlation(data):   
    
    # Golas: 검토 (Peak 위치 찾기)

    cross_correlation=data['cross_correlation']
    # 크로스 코릴레이션 데이터에 대해 봉우리를 찾습니다.
    # find_peaks 함수는 높이, 거리, 임계값 등을 파라미터로 사용할 수 있습니다.
    peaks, _ = find_peaks(cross_correlation, height=30,prominence=30)  # height=0은 모든 양의 봉우리를 찾는데 사용됨

    # 결과 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(cross_correlation, label='Cross-correlation')
    plt.scatter(peaks, cross_correlation[peaks], color='red', label='Detected Peaks')
    plt.title('Cross-correlation with Detected Peaks')
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation Coefficient')
    plt.legend()
    plt.show()
