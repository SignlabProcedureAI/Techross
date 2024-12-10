
# basic
import numpy as np

# visualization
import matplotlib.pyplot as plt

# module.healthchecker
import stat_healthchecker.rate_of_change_algorithms as rate_algorithms


def extract_rate_of_change_threshold(data):
    # 양의 변화율과 음의 변화율이 특정 임계값을 초과하는 지점을 찾습니다.
    positive_threshold = data['TRO_Ratio'].quantile(0.95)  # 상위 5%를 초과하는 변화율을 임계값으로 설정합니다.
    negative_threshold = data['TRO_Ratio'].quantile(0.05)  # 하위 5%를 초과하는 변화율을 임계값으로 설정합니다.
    
    #positive_threshold = data['TRO_Ratio']>=4  # 상위 5%를 초과하는 변화율을 임계값으로 설정합니다.
    #negative_threshold = data['TRO_Ratio']<=-4  # 하위 5%를 초과하는 변화율을 임계값으로 설정합니다.
    

    # 임계값을 초과하는 지점을 찾습니다.
    data['Peak'] = (data['TRO_Ratio'] > positive_threshold) & (data['TRO_Ratio']>=2)
    data['Valley'] = (data['TRO_Ratio'] < negative_threshold) & (data['TRO_Ratio']<=-2)
    
    return data


def plot_hunting_threshold(data):
    
    # 결과를 시각화합니다.
    plt.figure(figsize=(12, 6))
    plt.plot(data['TRO'], label='TRO')
    plt.scatter(data[data['Peak']].index, data[data['Peak']]['TRO'], color='green', label='Peaks', zorder=5)
    plt.scatter(data[data['Valley']].index, data[data['Valley']]['TRO'], color='red', label='Valleys', zorder=5)
    plt.title('Signal with Change Rate Peaks and Valleys')
    plt.legend()
    plt.show()

    # 변화율 임계값을 출력합니다.
    #print("Positive Change Rate Threshold:", positive_threshold)
    #print("Negative Change Rate Threshold:", negative_threshold)


# '헌팅' 라벨을 부여하는 함수를 정의합니다.
def label_hunting_multiple_of_two(df):
    # 변화율 계산
    df=rate_algorithms.calculating_rate_change(df,'TRO')
    
    # 변화율 임계 값 라벨링
    df=extract_rate_of_change_threshold(df)
    
    df['Hunting'] = 0  # 초기에는 모든 지점에 대해 '헌팅'이 아님을 가정합니다.
    peaks_indices = df[df['Peak']].index
    valleys_indices = df[df['Valley']].index
    
 # 피크와 밸리가 번갈아 나타나는지 확인합니다.
    sequence = []  # 번갈아 나타나는 피크와 밸리의 인덱스를 저장할 리스트

    # 피크와 밸리 인덱스를 번갈아가며 확인합니다.
    for index in np.sort(np.concatenate((peaks_indices, valleys_indices))):
        if not sequence:
            # sequence가 비어있으면 시작점으로 추가합니다.
            sequence.append(index)
        elif (sequence[-1] in peaks_indices and index in valleys_indices) or \
             (sequence[-1] in valleys_indices and index in peaks_indices):
            # 번갈아 나타나는 경우 sequence에 추가합니다.
            sequence.append(index)
        else:
            # 같은 타입이 연속해서 나타나는 경우 sequence를 유지합니다.
            continue

        # sequence 길이가 2의 배수이고 최소 4 이상인 경우, '헌팅' 라벨을 부여합니다.
        if len(sequence) >= 6 and len(sequence) % 3 == 0:
            df.at[sequence[-1], 'Hunting'] = 1
            sequence = []  # '헌팅' 라벨을 부여한 후 sequence를 리셋합니다.

    return df


def plot_hunting_label(df):
    # 결과를 시각화합니다.
    plt.figure(figsize=(12, 6))
    plt.plot(df['TRO'], label='TRO')
    plt.scatter(df[df['Peak']].index, df[df['Peak']]['TRO'], color='green', label='Peaks', zorder=5)
    plt.scatter(df[df['Valley']].index, df[df['Valley']]['TRO'], color='red', label='Valleys', zorder=5)
    plt.scatter(df[df['Hunting']==1].index, df[df['Hunting']==1]['TRO'], color='purple', label='Hunting', zorder=5, s=100)
    plt.title('Signal with Alternating Peaks, Valleys, and Hunting Label')
    plt.legend()
    plt.show()

    # '헌팅' 라벨이 있는 인덱스를 출력합니다.
    hunting_indices = df[df['Hunting']==1].index.tolist()
    print("Indices with 'Hunting' label:", hunting_indices)