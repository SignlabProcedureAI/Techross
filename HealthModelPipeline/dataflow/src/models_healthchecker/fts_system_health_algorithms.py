# basic
import numpy as np
import pandas as pd
import warnings
import os
import sys
import pickle

# module.healthchecker
import models_healthchecker.rate_of_change_algorithms as rate_algorithms

# module.dataline
from models_dataline.load_database import load_database


# set
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',2400)


def catorize_health_score(data):
    
    data['DEFECT_RISK_CATEGORY'] = 0

    data.loc[data['HEALTH_SCORE']<=13, 'RISK'] = 'NORMAL'
    data.loc[(data['HEALTH_SCORE']>13) & (data['HEALTH_SCORE']<=40), 'RISK'] = 'WARNING'
    data.loc[(data['HEALTH_SCORE']>40) & (data['HEALTH_SCORE']<=90), 'RISK'] = 'RISK'
    data.loc[data['HEALTH_SCORE']>90, 'RISK'] = 'DEFECT'

    return data


def refine_frames(data):
    """ 데이터 정제 함수
    """
    # 1. 컬럼 설정
    columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE','START_TIME','END_TIME','RUNNING_TIME']
    
     # 2.변수 선택
    data = data[columns]
    
    return data


def apply_system_health_statistics_with_fts(data):

    """ 그룹 통계 함수
    """
    # 1.데이터 그룹화

    data['DATA_TIME'] = pd.to_datetime(data['DATA_TIME'])
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # 시간 추출
    start_date = data.iloc[0,16]
    end_date = data.iloc[0,17]
    running_time = data.iloc[0,18]
    op_type = data.iloc[0,3]

    data = data[['SHIP_ID','OP_INDEX','DATA_INDEX','SECTION','CSU','STS','FMU','CURRENT','TRO','FTS','DIFF','THRESHOLD',
           'HEALTH_RATIO','HEALTH_TREND']]

    group = data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg({'DATA_INDEX':'mean','CSU':'mean','STS':'mean','FMU':'mean','CURRENT':'mean','TRO':'mean','FTS':['min','mean','max'],'DIFF':['min','mean','max'],'THRESHOLD':'mean','HEALTH_RATIO':'mean','HEALTH_TREND':'mean'})
    
    # 다중 인덱스된 컬럼을 단일 레벨로 평탄화
    group.columns = ['_'.join(col) for col in group.columns]
    group.columns = ['DATA_INDEX','CSU','STS','FMU','CURRENT','TRO','FTS_MIN','FTS_MEAN','FTS_MAX','DIFF_MIN','DIFF_MEAN','DIFF_MAX','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']

    # 2. Score 변수 추가
    score, trend_score = calculate_group_health_score(data,'FTS')

    group['HEALTH_SCORE'] = score
    group['TREND_SCORE'] = trend_score

    # 3. 데이터 인덱스 리셋
    group = group.reset_index()
    
    group['START_TIME'] =  start_date
    group['END_TIME'] = end_date
    group['RUNNING_TIME'] = running_time
    group['OP_TYPE'] = op_type 
    

    # 4, 변수 선택
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','CSU','STS','FMU','CURRENT','TRO','FTS_MIN','FTS_MEAN','FTS_MAX','DIFF_MIN','DIFF_MEAN','DIFF_MAX','THRESHOLD','TREND_SCORE',
           'HEALTH_RATIO','HEALTH_TREND','HEALTH_SCORE','START_TIME','END_TIME','RUNNING_TIME']]
    
    # 모델 로드

    # 현재 파일의 경로를 기준으로 model 폴더 내 fts_model 경로 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fts_model_relative_path = os.path.join(current_dir,".." , "models_model","fts_model")
    # 상대 경로를 절대 경로로 변환
    fts_model_path = os.path.abspath(fts_model_relative_path)

    model = load_model_from_pickle(fts_model_path)

    # 변수 선택 및 예측
    X = group[['FTS_MEAN','FTS_MAX','DIFF_MEAN','DIFF_MAX','TREND_SCORE']]
    pred =  model.predict(X)

    group['PRED'] = pred

    # 학습 데이터 적재
    # load_database('ecs_test','tc_ai_fts_system_health_group', group)

    # 웹 표출을 위한 변수 정제
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','HEALTH_SCORE','PRED','START_TIME','END_TIME','RUNNING_TIME']]
    group = catorize_health_score(group)
    group = group.rename({'HEALTH_SCORE':'ACTUAL'}, axis=1)

    group['ACTUAL'] = np.round(group['ACTUAL'],2)
    group['PRED'] = np.round(group['PRED'],2)

    # 뷰 데이터 적재
    # load_database('signlab','tc_ai_fts_model_system_health_group', group)
    
    return group


def exceed_limit_line(data,col):
    """
    해당 변수 임계 값 초과 시 해당 값 리턴
    """
    limit={'CSU': 45,
        'STS' : 40,
        'FTS' : 40} 
    # 변수 선택
    val = data[col].max()
    
    if val > limit[col]:
        return 100
    else :
        return 0


def generate_health_score(data,col):
    """
    건강도 함수 적용
    """
    
    # 1. 변화율 함수 샤용
    pre = rate_algorithms.calculating_rate_change(data,col)
    
    # 2. 결측치 제거
    pre.dropna(inplace=True)
    
    # 3. 임계치 설정
    pre['THRESHOLD'] = 0.22
    
    # 4. 변화율 절대 값
    pre[f'{col}_Ratio'] = abs(pre[f'{col}_Ratio'])
    
    # 5. 시스템 건강도 산정
    pre['HEALTH_RATIO'] = abs(pre[f'{col}_Ratio'] / pre['THRESHOLD']) * 10
    
    # 6. 이동평균 생성
    pre=rate_algorithms.generate_rolling_mean(pre,col,5)
    
    # 7. 변수명 변경
    pre.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE',
                 'START_TIME','END_TIME','RUNNING_TIME','FTS_Ratio','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    # 8. 변수명 재 설정
    pre = pre.rename({'FTS_Ratio':'DIFF'},axis=1)
    
    # 8. 범위 설정
    pre['HEALTH_RATIO'] = pre['HEALTH_RATIO'].apply(lambda x : 100 if x >= 100 else x)
    
    return pre



def calculate_group_health_score(data,col):
    """ 해당 센서 추세를 이용한 건강도 점수 반영
    """
    
     # . DATA_INDEX >= 30 이상(30분 이상 진행 된 데이터 필터링)
    data = data[data['DATA_INDEX']>=30]
    
    threshold = {'CSU': 0.88,
        'STS' : 1.18,
              'FTS': 1.75}
    
    first = data.iloc[0][col]
    last = data.iloc[-1][col]

    trend_score = (abs(last - first) / threshold[col]) * 10

    health_score = data['HEALTH_RATIO'].max()
    
    total_score = health_score + trend_score
    
    limit_score = exceed_limit_line(data,col)
    
    
    return [ 100 if total_score + limit_score >= 100 else total_score + limit_score], trend_score


# In[122]:


# Action: 알고리즘 적용

def apply_system_health_algorithms_with_fts(data):
    """ FTS 알고리즘 적용
    Args : 선박 이름, 오퍼레이션 번호, 섹션 번호
    
    Returns : 오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
    """
    
    # 1. 데이터 정제
    data = refine_frames(data)
    
    # 2. 시스텀 건강도 데이터 셋 구축
    system_data = generate_health_score(data,'FTS')

    # 3. 데이터 프레임 재 설정
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FMU','CURRENT','TRO','FTS','DIFF',
                    'THRESHOLD','HEALTH_RATIO','HEALTH_TREND','START_TIME','END_TIME','RUNNING_TIME']
    system_data_condition = system_data[position_columns]
                                                    
    
    # 5. 자동 적재
    # load_database()
    
    # 6. 그룹 적재 
    group = apply_system_health_statistics_with_fts(system_data_condition)

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