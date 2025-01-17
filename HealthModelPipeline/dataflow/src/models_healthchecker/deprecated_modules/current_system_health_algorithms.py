# basic
import numpy as np
import pandas as pd
import os
import pickle

# module.dataline
from models_dataline.load_database import load_database

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
    #start_val = 30
    #end_val = 50

    for i in range(index_start, index_end + 1):
        data.at[i, col] = start_val + ((end_val - start_val) * (i - index_start)) / (index_end - index_start)
        
    return data


def merge_tons(data):   
    # 데이터 로드
    tons = pd.read_csv(r"C:\Users\pc021\Desktop\프로젝트\테크로스\모델 고도화\건강도 데이터\data\tons.csv")
    
    # 데이터 
    tons = tons[['SHIP_ID','SECTION','tons']]
    
    tons.columns = ['SHIP_ID','SECTION','TONS']
    
    # Goals: 톤수 병합
    merge = pd.merge(data,tons,on=['SHIP_ID','SECTION'],how='left')
    
    return merge


def catorize_health_score(data):
    
    data['DEFECT_RISK_CATEGORY'] = 0

    data.loc[data['ELECTRODE_EFFICIENCY']>=-16, 'RISK'] = 'NORMAL'
    data.loc[(data['ELECTRODE_EFFICIENCY']<-16) & (data['ELECTRODE_EFFICIENCY']>=-40), 'RISK'] = 'WARNING'
    data.loc[(data['ELECTRODE_EFFICIENCY']<-40) & (data['ELECTRODE_EFFICIENCY']>=-90), 'RISK'] = 'RISK'
    data.loc[data['ELECTRODE_EFFICIENCY']<-90, 'RISK'] = 'DEFECT'

    return data

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
    data['GENERALIZATION_EFFICIENCY'] = (100*data['TRO'])/((1.323*data['CURRENT']) / data['FMU']) * (1 + k * (1 - data['CSU'] / 50))
    # data['GENERALIZATION_EFFICIENCY'] = (100*data['TRO'])/((1.323*data['CURRENT']) / data['FMU']) 
    
    return data


def apply_system_health_statistics_with_current(data):
    """ CURRENT 그룹 알고리즘 적용
    """
    
  
    data['DATA_TIME'] = pd.to_datetime(data['DATA_TIME'])
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # 시간 추출
    start_date = data.iloc[0,15]
    end_date = data.iloc[0,16]
    running_time = data.iloc[0,17]
    op_type = data.iloc[0,3]
    
    
    # 데이터 선택
    data = data[data['DATA_INDEX']>=30]
        
    # 데이터 그룹화
    group = data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()[['CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY']]
    
    # 데이터 인덱스 리셋
    group = group.reset_index()
    
    group['START_TIME'] =  start_date
    group['END_TIME'] = end_date
    group['RUNNING_TIME'] = running_time
    group['OP_TYPE'] = op_type 
    
    # 컬럼 재 정렬
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','START_TIME','END_TIME','RUNNING_TIME','CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY']]
    
    # 학습 데이터 적재
    load_database('ecs_test','tc_ai_electrode_group_v1.1.0', '200', group)

    # 모델 로드

    # 현재 파일의 경로를 기준으로 model 폴더 내 ecu_model 경로 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ecu_model_relative_path = os.path.join(current_dir,".." , "models_model","ecu_model_v2.0.0")
    # 상대 경로를 절대 경로로 변환
    ecu_model_path = os.path.abspath(ecu_model_relative_path)

    model = load_model_from_pickle(ecu_model_path)

    # 변수 선택 및 예측
    X = group[['CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT']]
    pred =  model.predict(X)

    group['PRED'] = pred

    # 웹 표출을 위한 변수 정제
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','ELECTRODE_EFFICIENCY','PRED','START_TIME','END_TIME','RUNNING_TIME']]
    group = catorize_health_score(group)
    group = group.rename({'ELECTRODE_EFFICIENCY':'ACTUAL'}, axis=1)

    group['ACTUAL'] = np.round(group['ACTUAL'],2)
    group['PRED'] = np.round(group['PRED'],2)

    # 뷰 데이터 적재
    load_database('signlab','tc_ai_electrode_model_group', 'release', group)
    # load_database('ecs_test','tc_ai_electrode_model_group_flag', '200', group)

    return group


def apply_system_health_algorithms_with_current(data):
    """ CURRENT 알고리즘 적용
        Args : 
          선박 이름, 오퍼레이션 번호, 섹션 번호
    
        Returns :
          오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
    """
    
    # 1. 데이터 정제
    data = refine_frames(data)

    # 2. 효율식을 이용한 건강도 지표 생성
    system_data = calculate_generalization_value(data)

    # 3. 효율 값을 마이너스 지표로 변경
    system_data = calculate_minus_value(system_data)

    # 4. 변수 선택
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']                      
    system_data_condition = system_data[position_columns]                                               

    # 5. 변수명 변경
    system_data_condition.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']

    # 6. 자동 적재
    # load_database()

    # 7. 그룹 적재 
    system_data_condition = system_data_condition[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME']]
    group = apply_system_health_statistics_with_current(system_data_condition)

    return system_data_condition, group
    

def load_model_from_pickle(file_path):
    """
    피클 파일에서 모델을 불러오는 함수.

    Args:
    - file_path: 불러올 피클 파일의 경로

    Returns:
    - model: 불러온 모델 객체
    """
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"모델이 {file_path}에서 성공적으로 불러와졌습니다.")
    return model