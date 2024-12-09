#!/usr/bin/env python
# coding: utf-8
import sys
import os

# model_upgrading_path = os.path.join("..","src")
# sys.path.append(model_upgrading_path)

# basic
import numpy as np
import pandas as pd
import pickle

# module.algorithms
import update_package.algorithms.rate_of_change_algorithms as rate_algorithms

# module.data
from update_package.data.load_database import load_database

# set
import warnings
warnings.filterwarnings("ignore")


# In[83]:

def catorize_health_score(data):
    
    data['DEFECT_RISK_CATEGORY'] = 0

    data.loc[data['HEALTH_SCORE']<=14, 'RISK'] = 'NORMAL'
    data.loc[(data['HEALTH_SCORE']>15) & (data['HEALTH_SCORE']<=40), 'RISK'] = 'WARNING'
    data.loc[(data['HEALTH_SCORE']>40) & (data['HEALTH_SCORE']<=90), 'RISK'] = 'RISK'
    data.loc[data['HEALTH_SCORE']>90, 'RISK'] = 'DEFECT'

    return data

def refine_frames(data):
    # 변수 선택
    columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE','VOLTAGE','START_TIME','END_TIME','RUNNING_TIME']
    data = data[columns]
    
    return data


# In[111]:


# Goals: 그룹 통계 함수 적용
def apply_system_health_statistics_with_csu(data):

    data['DATA_TIME'] = pd.to_datetime(data['DATA_TIME'])
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # 시간 추출
    start_date = data.iloc[0,16]
    end_date = data.iloc[0,17]
    running_time = data.iloc[0,18]
    op_type = data.iloc[0,3]

    data = data[['SHIP_ID','OP_INDEX','DATA_INDEX','SECTION','STS','FTS','FMU','CURRENT','TRO','CSU','DIFF','THRESHOLD',
           'HEALTH_RATIO','HEALTH_TREND']]


    # 데이터 그룹화
    group = data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg({'DATA_INDEX':'mean','STS':'mean','FTS':'mean','FMU':'mean','CURRENT':'mean','TRO':'mean','CSU':['min','mean','max'],'DIFF':['min','mean','max'],'THRESHOLD':'mean','HEALTH_RATIO':'mean','HEALTH_TREND':'mean'})
    
    # 다중 인덱스된 컬럼을 단일 레벨로 평탄화
    group.columns = ['_'.join(col) for col in group.columns]
    group.columns = ['DATA_INDEX','STS','FTS','FMU','CURRENT','TRO','CSU_MIN','CSU_MEAN','CSU_MAX','DIFF_MIN','DIFF_MEAN','DIFF_MAX','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']

    # SCORE 변수 추가
    score, trend_score = calculate_group_health_score(data,'CSU')
    
    group['HEALTH_SCORE'] = score
    group['TREND_SCORE'] = trend_score

    # 데이터 인덱스 리셋
    group = group.reset_index()
    
    group['START_TIME'] =  start_date
    group['END_TIME'] = end_date
    group['RUNNING_TIME'] = running_time
    group['OP_TYPE'] = op_type 
    
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','STS','FTS','FMU','CURRENT','TRO','CSU_MIN','CSU_MEAN','CSU_MAX','DIFF_MIN','DIFF_MEAN','DIFF_MAX','THRESHOLD','TREND_SCORE',
           'HEALTH_RATIO','HEALTH_TREND','HEALTH_SCORE','START_TIME','END_TIME','RUNNING_TIME']]
    
    # 모델 로드

    # 현재 파일의 경로를 기준으로 model 폴더 내 csu_model 경로 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csu_model_relative_path = os.path.join(current_dir,".." , "model","csu_model")
    # 상대 경로를 절대 경로로 변환
    csu_model_path = os.path.abspath(csu_model_relative_path)

    model = load_model_from_pickle(csu_model_path)

    # 변수 선택 및 예측
    X = group[['CSU_MIN', 'CSU_MEAN', 'CSU_MAX', 'DIFF_MIN', 'DIFF_MEAN', 'DIFF_MAX', 'TREND_SCORE']]
    pred =  model.predict(X)

    group['PRED'] = pred

    # 학습 데이터 적재
    load_database('ecs_test','tc_ai_csu_system_health_group', group)

    # 웹 표출을 위한 변수 정제
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','HEALTH_SCORE','PRED','START_TIME','END_TIME','RUNNING_TIME']]
    group = catorize_health_score(group)
    group = group.rename({'HEALTH_SCORE':'ACTUAL'}, axis=1)
    
    group['ACTUAL'] = np.round(group['ACTUAL'],2)
    group['PRED'] = np.round(group['PRED'],2)

    # 뷰 데이터 적재
    load_database('signlab','tc_ai_csu_model_system_health_group', group)
    
    return group


# In[102]:


def exceed_limit_line(data,col):
    limit={'CSU': 45,
        'STS' : 40}
    
    # 변수 선택
    val = data[col].max()
    
    if val > limit[col]:
        return 100
    else :
        return 0


# In[103]:


def generate_health_score(data,col):
    
    # 변화율 함수 샤용
    pre = rate_algorithms.calculating_rate_change(data,col)
    
    # Nul; 값 제거
    pre.dropna(inplace=True)
    
    # 임계치 설정
    pre['THRESHOLD'] = 0.18
    
    # 변화율 절대 값
    pre[f'{col}_Ratio'] = abs(pre[f'{col}_Ratio'])
    
    # 시스템 건강도 산정
    pre['HEALTH_RATIO'] = abs(pre[f'{col}_Ratio'] / pre['THRESHOLD']) * 10
    
    # 이동평균 생성
    pre = rate_algorithms.generate_rolling_mean(pre,col,5) 
    
    # 변수명 변경
    pre.columns=['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','RATE', 'VOLTAGE',
                 'START_TIME','END_TIME','RUNNING_TIME','CSU_Ratio','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    pre = pre.rename({'CSU_Ratio':'DIFF'},axis=1)
    
    # 범위 설정
    pre['HEALTH_RATIO']=pre['HEALTH_RATIO'].apply(lambda x : 100 if x >= 100 else x)
    
    
    return pre


# In[104]:


def calculate_group_health_score(data,col):
    threshold={'CSU': 0.88,
        'STS' : 1.18}
    
    first = data.iloc[0][col]
    last = data.iloc[-1][col]

    trend_score = (abs(last - first) / threshold[col]) * 10

    health_score = data['HEALTH_RATIO'].max()
    
    total_score = health_score + trend_score
    
    limit_score = exceed_limit_line(data,col)
    
    return [ 100 if total_score + limit_score >= 100 else total_score + limit_score],trend_score


# In[105]:


# Golas: 알고리즘 적용

def apply_system_health_algorithms_with_csu(data):
    """ 인자: 선박 이름, 오퍼레이션 번호, 섹션 번호
    
        반환 값: 오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
    """
    
    # 1. 데이터 정제
    data = refine_frames(data)
    
    # 2. 시스텀 건강도 데이터 셋 구축
    system_data = generate_health_score(data,'CSU')

    # 3. 데이터 프레임 재 설정
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','STS','FTS','FMU','CURRENT','TRO','CSU','DIFF',
                    'THRESHOLD','HEALTH_RATIO','HEALTH_TREND','START_TIME','END_TIME','RUNNING_TIME']
    system_data_condition = system_data[position_columns]                                          
    
    # 4. 자동 적재
    # load_database(system_data_condition,'test_tc_ai_csu_system_health')
    
    # 5. 그룹 적재 
    group = apply_system_health_statistics_with_csu(system_data_condition)

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

