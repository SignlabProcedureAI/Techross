import pickle
import numpy as np
import pandas as pd

def generate_rolling_mean(data,col,window_size):
    
    # 데이터 정리
    data.sort_index(inplace=True)
    
    # 이동 평균 생성
    data['rolling_mean']=data[col].rolling(window=window_size).mean()
    
    # 결측치 제거
    data.dropna(inplace=True)
    
    # 정제된 함수 반환 
    return data


def load_pickle(file_path):
    """
    주어진 파일 경로에서 피클 파일을 로드하는 함수.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        #print(f"Loaded data type: {type(data)}")  # 로드된 데이터 타입 출력
        return data
    except Exception as e:
        #print(f"Error loading pickle file: {e}")
        return None
  

def calculating_rate_change_std(df, column, std):
    """
    CSU std: 1.78
    FMU std: 351 
    FTS std: 2.09
    TRO std: 0.75
    CURRENT std: 2287
    
    """
    
    # 지정된 열의 데이터를 추출합니다.
    data = df[column]

    # 이전 값과의 차이를 계산합니다.
    differences = data - data.shift(1)

    # 데이터의 표준편차를 계산합니다.
    standard_deviation = std

    # 차이를 표준편차로 나누어 정규화합니다.
    normalized = np.round((differences / standard_deviation), 2)

    # 계산된 비율을 새 열로 추가합니다.
    df['{}_Ratio'.format(column)] = normalized

    return df


def calculating_rate_change_exclude_op_index(df, column):
    # 변경 지점을 식별합니다.
    change_indices = df[df['OP_INDEX'] != df['OP_INDEX'].shift(1)].index

    # 변경 지점의 값을 NaN으로 설정합니다.
    df.loc[change_indices, column] = np.nan
    
    # 이전 값과의 차이를 계산합니다.
    differences = df[column] - df[column].shift(1)

    # 계산된 비율을 새 열로 추가합니다.
    df['{}_Ratio'.format(column)] = differences

    return df


def calculating_rate_change(df, column):
  
    # 이전 값과의 차이를 계산합니다.
    differences = df[column] - df[column].shift(1)

    # 계산된 비율을 새 열로 추가합니다.
    df['{}_Ratio'.format(column)] = np.abs(differences)

    return df


def generate_tro_neg_count(abnormal):
    """ 음수 카운트를 저장할 새로운 열을 초기화합니다.
    """
    
    abnormal['TRO_NEG_COUNT'] = 0
    negative_count = 0
   
    def calculate_true_difference(tro_value, predicted_value):
        return np.abs(tro_value - predicted_value)
    
    def is_decline(tro_value, predicted_value,temporary_increase_threshold):
        true_diff = calculate_true_difference(tro_value, predicted_value)
        return  (tro_value < temporary_increase_threshold) and (tro_value < predicted_value) and (true_diff > 0.24)

    # 간헐적 증가 확인 함수
    def has_temporary_increase(tro_value):
        temporary_increase_threshold=0.13
        if (tro_value > temporary_increase_threshold) :
            return True
        return False

    def negative_has_temporary_increase(tro_value,predicted_value):
        temporary_increase_threshold=-0.13
        if is_decline(tro_value,predicted_value,temporary_increase_threshold):
            return True
        return False


    for i in range(len(abnormal)):
        tro_value = abnormal.iloc[i, abnormal.columns.get_loc('TRO_Ratio')]
        predicted_value=abnormal.iloc[i, abnormal.columns.get_loc('pred_Ratio')]
    
        # 현재 값이 음수인 경우
        temporary_increase_check=negative_has_temporary_increase(tro_value,predicted_value)
        if tro_value < -1.5 and temporary_increase_check: 
            negative_count += 1  # 음수의 카운트를 증가
        else:
            temporary_increase=has_temporary_increase(tro_value)
            if tro_value> 0 and temporary_increase:
            # 현재 값이 양수인 경우, 카운트 리셋
                negative_count = 0

        # 음수 카운트 업데이트 (양수가 나오면 0으로 리셋되고, 음수가 나오면 카운트 증가, 0은 카운트에 영향을 주지 않음)
        abnormal.iloc[i, abnormal.columns.get_loc('TRO_NEG_COUNT')] = negative_count
        
    return abnormal


    
def calculate_true_difference(delta, predicted_delta):
    return np.abs(delta - predicted_delta)

def is_steep_decline(delta, predicted_delta):
    true_diff = calculate_true_difference(delta, predicted_delta)
    #return delta <= -1.2 and (delta < predicted_delta) and (true_diff > 0.24)
    #return delta <= -3 and (delta < predicted_delta) and (true_diff > 0.24)
    return delta <= -5 and (delta < predicted_delta) and (true_diff > 0.24)

def classify_decline_steep_label(delta, predicted_delta):
    if is_steep_decline(delta, predicted_delta):
        # Consider using logging instead of print for real-world applications
        #return 'Steep Decline'
        return 1
    else:
        # Consider using logging instead of print for real-world applications
        #return 'Normal'
        return 0
    
    
def is_slowly_decline(pre_neg_count,neg_count):
    return (neg_count% 3 == 0) & (neg_count != 0) & (pre_neg_count!=neg_count)

def classify_decline_slowly_label(pre_neg_count,neg_count):
    if is_slowly_decline(pre_neg_count,neg_count):
        # Consider using logging instead of print for real-world applications
        #return 'Slowly Decreasing'
        return 1
    else:
        # Consider using logging instead of print for real-world applications
        #return 'Normal'     
        return 0
 
    

def give_tro_condition(data):
    # TRO 조건 부여
    data.loc[data['TRO'] >= 8,'STEEP_LABEL'] = 0
    
    return data
    
  
def give_tro_out_of_water_condition(data):
    data['OUT_OF_WATER_STEEP'] = 0
    # 조건 부여
    data.loc[(data['TRO'] <= 1) & (data['STEEP_LABEL']==1) ,'OUT_OF_WATER_STEEP'] = 1
    data.loc[(data['TRO'] <= 1) & (data['STEEP_LABEL']==1) ,'STEEP_LABEL'] = 0
    
    return data