# basic
import pandas as pd

# visualize
import seaborn as sns
import matplotlib.pyplot as plt

# moduel.dataline
from stat_dataline.select_dataset import get_dataframe_from_database
from stat_dataline.load_database import load_database

# set
import warnings
warnings.filterwarnings('ignore')



# Action: 이동평균 함수 구현
def extract_moving_average(data, window_size):
    
    # 이동평균 계산
    moving_avg = data.rolling(window=window_size).mean()
    
    # NaN 값을 첫 번째 유효한 값으로 대체
    first_valid_value = moving_avg.dropna().iloc[0]
    moving_avg.fillna(first_valid_value, inplace=True)
    
    return moving_avg



# Action: 이동평균 함수 구현
# 데이터 로드
data = get_all_dataframe_from_database('tc_ai_total_system_health_group')

# 데이터 프레임을 담을 리스트
total_list = []

# 배 이름 추출
unique_ships = data['SHIP_ID'].unique()

# 반복문을 활용한 이동평균 생성
for unique_ship in unique_ships:
    
    # 특정 선택
    selected_data = data[(data['SHIP_ID']==unique_ship)] 
    
    # 변수 선택
    selected_val_data = selected_data[['SHIP_ID','OP_INDEX','SECTION','TOTAL_HEALTH_SCORE']]
    
    # 데이터 프레임 길이 추출
    length = len(selected_val_data)
    
    # 만약 길이가 3이상일 경우
    if length>=3:
        # 이동평균 함수 사용
        moving_avg = extract_moving_average(selected_val_data['TOTAL_HEALTH_SCORE'],3)
        
        # 생성된 이동평균 추가
        selected_val_data['TREND'] = moving_avg
        
        # 데이터 프레임 리스트 추가
        total_list.append(selected_val_data)

# 데이터 프레임 결합
loaded_dataframe = pd.concat(total_list)

# 데이터 프레임 적재
load_database(loaded_dataframe,'tc_ai_total_system_health_group_trend')


# In[45]:


def plot_system_health(data):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='OP_INDEX',y='TREND',data=data,label='Health TREND', color='blue',linestyle='--')
    #sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health Trend', color='orange',linestyle='--')
    
     # 모든 x 위치에 'X' 표시 추가
    #plt.scatter(data['DATA_INDEX'], data['HEALTH_TREND'], marker='x', color='red', label='All Points')
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Comprehensive System Health Trend')
    plt.ylim(0,100)
    plt.xlabel('Operation Index')
    plt.ylabel('Health Trend')
    
    # 플롯 표시
    plt.show()
    
