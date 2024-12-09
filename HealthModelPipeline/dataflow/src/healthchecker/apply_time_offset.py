#!/usr/bin/env python
# coding: utf-8

# In[4]:

# basic
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:

# math
from scipy.signal import correlate
from scipy.signal import find_peaks,periodogram
from scipy.fft import fft


# In[79]:

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
    
    data = apply_threshold(data)
    
    corrlat = data['cross_correlation'].values

    autocorr = correlate(corrlat, corrlat, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    #data = check_signal(data)
    result, count = filter_repeating_extrema(autocorr)
    data = evaluate_time_offset(result,data)
    
    data = limit_date_time(data)
    
    data = data.rename({'cross_correlation':'CROSS_CORRELATION','lag':'LAG'},axis=1)

    data['RE_CROSS_CORRELATION'] = autocorr
    
    return data, count



# In[104]:


def apply_thresholdb(data):
    
    cross_max=data['cross_correlation'].max()
    
    # 크로스 코릴레이션 값이 50 이상인 지점의 인덱스를 찾음
    threshold = cross_max/2

    positive_threshold = data['cross_correlation'].quantile(0.95)  # 상위 5%를 초과하는 변화율을 임계값으로 설정합니다.
    
    # TIME OFFSET 변수를 데이터 프레임에 추가하고 모든 값을 False로 초기화
    data['TIME_OFFSET'] = 0

    # 조정된 lags가 원래 데이터의 인덱스 범위 내에 있는지 확인하고,
    # 해당 인덱스에 대해 TIME OFFSET 값을 True로 설정
     # 임계값을 초과하는 지점을 찾습니다.
    data['Peak'] = (data['cross_correlation'] > positive_threshold) & (data['cross_correlation']>=threshold)
    
    peaks_indices = data[data['Peak']].index
    
    
    #data.loc[data['cross_correlation']>=threshold,'TIME_OFFSET']=1
    
    data.loc[peaks_indices,'TIME_OFFSET']=1
        
    return data

def apply_threshold(data):
    #data['cross_correlation'] = savgol_filter(data['cross_correlation'], window_length=51,polyorder=3)
    
    cross_correlation=data['cross_correlation']
    
    cross_max = cross_correlation.max()
    
    threshold = (cross_max / 2)
    
    peaks, rest = find_peaks(cross_correlation, height=30,  prominence=30)  # height=0은 모든 양의 봉우리를 찾는데 사용됨
    
    # TIME OFFSET 변수를 데이터 프레임에 추가하고 모든 값을 False로 초기화
    data['TIME_OFFSET'] = 0
    
    #for peak_index in peaks:
        #cross_correlation=data.at[peak_index,'cross_correlation']
        #if (cross_correlation > threshold) & (abs(cross_correlation) > threshold):
            #data.at[peak_index, 'TIME_OFFSET'] = 1
            
          
    for i in range(2,len(peaks),3):
        data.loc[peaks[i], 'TIME_OFFSET'] = 1
    
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
    
    #print(f'극대 값: {maxima}')
    #print(f'극소 값: {minima}')
    
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
                return True,count
            count = 1
            alternating = False

    # Final check in case the last segment meets the criteria
    if alternating and count >= repeat_threshold:
        return True,count

    return False,count


def evaluate_time_offset(result,data):
    
    if result:
        return data
    else:
        data['TIME_OFFSET'] = 0
        return data
    

# In[105]:
def check_signal(data):
    cross_correlation = data['cross_correlation']
    
    # 피크 찾기
    peaks, _ = find_peaks(cross_correlation)
    peak_heights = cross_correlation[peaks]

    # 피크 간격의 표준편차 계산
    peak_intervals = np.diff(peaks)
    interval_std = np.std(peak_intervals)

    # 주파수 분석 (푸리에 변환)
    frequencies, power = periodogram(cross_correlation)
    
    # 가장 강한 주파수 성분 찾기
    strongest_freq = frequencies[np.argmax(power)]
    strongest_power = np.max(power)

    if (strongest_power < 100000) :
        data['TIME_OFFSET']=0

    return data
    
def limit_date_time(data):
    data.loc[data['DATA_INDEX'] < 30,'TIME_OFFSET'] = 0
    
    return data

    
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

def plot_sample_corr_fft(data):
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
    
    peaks, _ = find_peaks(autocorr, height=upper_value,  prominence=upper_value/4, distance=len(autocorr)/10)
    re_peaks, re_ = find_peaks(-autocorr, height=-lower_value,  prominence=(-lower_value/4), distance=len(autocorr)/10)
    
    print(peaks)
    print(re_peaks)
    
    plt.plot(peaks, autocorr[peaks], "rx")
    plt.plot(re_peaks, autocorr[re_peaks], "rx")
        
    plt.title('Autocorrelation with Detected Peaks')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    
    plt.tight_layout()
    plt.show()

    
def plot_sample_corr_fft(data):
    """ 데이터는 자기 상관계수를 인풋
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
    
    peaks, _ = find_peaks(autocorr, height=upper_value,  prominence=upper_value/4, distance=len(autocorr)/10)
    re_peaks, re_ = find_peaks(-autocorr, height=-lower_value,  prominence=(-lower_value/4), distance=len(autocorr)/10)
    
    print(peaks)
    print(re_peaks)
    
    plt.plot(peaks, autocorr[peaks], "rx")
    plt.plot(re_peaks, autocorr[re_peaks], "rx")
        
    plt.title('Autocorrelation with Detected Peaks')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    
    plt.tight_layout()
    plt.show()