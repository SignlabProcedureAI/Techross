# basic
import numpy as np
import pandas as pd

# module.dataline
from stat_dataline.load_database import load_database

# set
import warnings
warnings.filterwarnings('ignore')


def generate_bad_data(data,col,start_val,end_val):
    # 데이터 인덱스 정리
    data.reset_index(drop=True,inplace=True)
    # 데이터 길이 
    length = len(data)
    
    # 값 조정
    index_start = int(length/2)
    index_end = length-1

    for i in range(index_start, index_end + 1):
        data.at[i, col] = start_val + ((end_val - start_val) * (i - index_start)) / (index_end - index_start)
        
    return data



def merge_tons(data):   
    # 데이터 로드
    tons = pd.read_csv(r"C:\Users\pc021\Desktop\프로젝트\테크로스\시스템건강도\data\tons.csv")
    
    # 데이터 
    tons = tons[['SHIP_ID','SECTION','tons']]
    
    # Goals: 톤수 병합
    merge = pd.merge(data,tons,on=['SHIP_ID','SECTION'],how='left')
    
    return merge



def refine_frames(data):
    # 변수 선택
    columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE','START_TIME','END_TIME','RUNNING_TIME']

    data = data[columns]
    
    return data



def calculate_minus_value(data):
    
    data['ELECTRODE_EFFICIENCY'] =  - (100 - data['GENERALIZATION_EFFICIENCY'])
    
    data['ELECTRODE_EFFICIENCY'] = data['ELECTRODE_EFFICIENCY'].apply(lambda x: 0 if x >= 0 else x)
    
    return data



def calculate_generalization_value(data):
    k = 0.23
    #data['GENERALIZATION_EFFICIENCY'] = (100*data['TRO'])/((1.323*data['CURRENT']) / data['FMU']) 
    data['GENERALIZATION_EFFICIENCY'] = (100*data['TRO'])/((1.323*data['CURRENT']) / data['FMU']) * (1 + k * (1 - data['CSU'] / 50))
    return data



def apply_system_health_statistics_with_current(data):

    """그룹 통계 함수 적용
    
    Args:
     
    
    Return: 
     
    """
    data['DATA_TIME'] = pd.to_datetime(data['DATA_TIME'])
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # 시간 추출
    start_date = data.iloc[0,7]
    end_date = data.iloc[0,8]
    running_time = data.iloc[0,9]
    op_type = data.iloc[0,3]
    #running_time = end_date - start_date
    #running_time_in_minutes  = running_time.total_seconds() / 60



    # 데이터 선택
    data = data[data['DATA_INDEX']>=30]
        
    # 데이터 그룹화
    group = data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()['ELECTRODE_EFFICIENCY']

    # 데이터 인덱스 리셋
    group = group.reset_index()

    group['START_TIME'] =  start_date
    group['END_TIME'] = end_date
    group['RUNNING_TIME'] = running_time
    group['OP_TYPE'] = op_type 
    
    # 데이터 적재
    load_database('test', 'tc_ai_current_system_health_group', group) 

    return group

# In[45]:


def apply_system_health_algorithms_with_current(data):
    """ CURRENT 건강도 알고리즘 적용 
    Args: 
     선박 이름, 오퍼레이션 번호, 섹션 번호
    
    Returns: 
     오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재

    """

    # 1. 데이터 정제
    data = refine_frames(data)
    
    # 2. 효율식을 이용한 건강도 지표 생성
    system_data = calculate_generalization_value(data)
    
    # 3 효율 값을 마이너스 지표로 변경
    system_data = calculate_minus_value(system_data)
    
    # 4. 변수 선택
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']                      
    system_data_condition=system_data[position_columns]                                               
    
    # 5. 변수명 변경
    system_data_condition.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']
    
    # 6. 변수 반올림
    system_data_condition['ELECTRODE_EFFICIENCY'] = np.round(system_data_condition['ELECTRODE_EFFICIENCY'],2)

    # 7. 그룹 적재 
    group = apply_system_health_statistics_with_current(system_data_condition)

    # 8. 분당 센서 데이터 변수 선택
    sensor_data = system_data_condition[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','ELECTRODE_EFFICIENCY']]

    # 9. 자동 적재
    load_database('test','tc_ai_current_system_health', sensor_data)

    return sensor_data, group
    

