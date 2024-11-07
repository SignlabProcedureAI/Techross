#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Handling
import numpy as np
import pandas as pd
import warnings
import joblib
import os

# module.algorithms
import my_package.algorithms.rate_of_change_algorithms as rate_algorithms

# module.data
from my_package.data.load_database import load_database

# Preprocessing # 
from sklearn.preprocessing import StandardScaler


# In[3]:


# Action: 경고 설정

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',2400)


# In[4]:


def normalize_series(data_series,ship_id):
    
    """ 표준화/정규화 함수
    """   
    # 1.StandardScaler 객체 로드

    fmu_scaler_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(fmu_scaler_path, '../../../../health_data/data/fmu_standard_scaler'))

    scaler =  joblib.load(fr'{data_dir}\\{ship_id}_scaler.joblib')
    
    # 2. 데이터 시리즈에 대해 표준화 수행
    standardized_data = scaler.transform(data_series)
    
    return standardized_data


# In[5]:


def refine_frames(data):
    """ 데이터 정제 함수
    """
    # 1. 컬럼 설정
    columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO']
    
     # 2.변수 선택
    data = data[columns]
    
    return data


# In[6]:


def apply_system_health_statistics_with_fmu(data):
    
    """ 그룹 통계 함수
    """
    # 1.데이터 그룹화
    group=data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
    
    # 2. Score 변수 추가
    score = calculate_group_health_score(data,'FMU')
    group['HEALTH_SCORE'] = score
    
    # 3. 데이터 인덱스 리셋
    group = group.reset_index()
    
    # 4, 변수 선택
    group = group[['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FMU','STANDARDIZE_FMU','HEALTH_SCORE']]
    
    # 5. 데이터 적재
    #load_database(group,'tc_ai_fmu_system_health_group')
    
    return group


# In[7]:


def calculate_group_health_score(data,col):
    """ 해당 센서 추세를 이용한 건강도 점수 반영
    """
    # 1. DATA_INDEX >= 30 이상(30분 이상 진행 된 데이터 필터링)
    filtered_data = data[data['DATA_INDEX']>=30]
    
    # 2. 건강도 최대 값 추출
    health_score = filtered_data['HEALTH_RATIO'].max()
    
    return health_score


# In[8]:


def generate_health_score(data,col,ship_id):
    """
    건강도 함수 적용
    """
    # 1. 정규화 함수 사용
    data['STANDARDIZE_FMU'] = normalize_series(data[[col]],ship_id)
    standarize_fmu = 'STANDARDIZE_FMU'
        
    # 1.5 변수 변경
    pre = data
    
    # 2. 임계치 설정
    pre['THRESHOLD'] = 1.96
    
    # 3. 변화율 절대 값
    pre[f'{standarize_fmu}'] = abs(pre[f'{standarize_fmu}'])
    
    # 4. 시스템 건강도 산정
    pre['HEALTH_RATIO'] = abs(pre[f'{standarize_fmu}'] / pre['THRESHOLD']) * 30
    
    # 5. 이동평균 생성
    pre = rate_algorithms.generate_rolling_mean(pre,col,5)
    
    # 6. 변수명 변경
    pre.columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
                'STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    # 7. 범위 설정
    pre['HEALTH_RATIO'] = pre['HEALTH_RATIO'].apply(lambda x : 100 if x >= 100 else x)
    
    return pre


# In[9]:


# Action: 알고리즘 적용

def apply_system_health_algorithms_with_fmu(data, ship_id):
    """ 인자: 선박 이름, 오퍼레이션 번호, 섹션 번호
    
        반환 값: 오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재
    """
    
    # 1. 데이터 정제
    data = refine_frames(data)
    
    # 2. 시스텀 건강도 데이터 셋 구축
    system_data = generate_health_score(data,'FMU', ship_id)

    # 3. 데이터 프레임 재 설정
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FMU',
                      'STANDARDIZE_FMU','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    # 4. 변수 선택
    system_data_condition = system_data[position_columns]
                                                   
    # 5. 그룹 적재 
    group = apply_system_health_statistics_with_fmu(system_data_condition)

    return system_data_condition,group

