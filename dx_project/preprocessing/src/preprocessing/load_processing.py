#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os

# 경로 설정: 스크립트 경로에서 상위 디렉토리로 이동한 후 src 경로 추가
health_data_path = os.path.abspath(os.path.join('..', 'src'))
health_learning_data_path = os.path.abspath(os.path.join(os.getcwd(), "../../health_learning_data/health_data/src"))
preprocessing_path = os.path.abspath(os.path.join(os.getcwd(), "../../preprocessing/src"))

paths = [health_data_path, health_learning_data_path, preprocessing_path]

def add_paths(paths):
    """
    지정된 경로들이 sys.path에 없으면 추가하는 함수.
    
    Parameters:
    - paths (list): 추가하려는 경로들의 리스트.
    """
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
            print(f"Path added: {path}")
        else:
            print(f"Path already exists: {path}")
            
add_paths(paths)

# module.algorithms
from preprocessing.preprocessing import apply_preprocessing_fuction
from visualization import visualize as visual

# module.data
from data.select_dataset_fluid import get_dataframe_from_database_fluid as get_dataframe
from data.select_dataset  import get_dataframe_from_database
from data.select_dataset_optime import get_dataframe_from_database_optime
from data.load_database import load_database
from my_package.data.logger_confg import logger

# basic
import json
import pandas as pd
from datetime import datetime
import time

# In[8]:


# Goals: 데이터 로드


# In[3]:


def select_data_variable(original_data, data):
    drop_col = ['vessel','tons','tons_category']
    
    original_data = original_data.drop(columns=drop_col)
    data = data.drop(columns=drop_col)
    
    original_data.rename({'ship_name':'SHIP_NAME'},axis=1)
    data = data.rename({'ship_name':'SHIP_NAME'},axis=1)
    
    return original_data,data


# In[4]:


# Goals: 데이터 추출 후 변수 적용

def distribute_variables(ship_id, op_index, section):
    
    # 데이터 베이스 등록 데이터 추출
    # get_data = get_dataframe('tc_data_preprocessing',ship_id, op_index, section) 테크로스 납품 db 접속 안되어 상수 설정

    # # 인덱스 추출
    # idx = get_data['PRE_INDEX'][0]

    # # 등록 시간 추출
    # reg = get_data['REG_DATE'][0]
    # reg = reg.strftime('%Y-%m-%d')

    # # op_type 추출
    # op_type = 'ba' if get_data['OP_TYPE'][0]!=2 else 'de'

    # BOA 정의
    #boa = 'before'
    
    # return idx,reg,op_type
    
    today = "2024-10-17"
    today_date = datetime.strptime(today, '%Y-%m-%d')

    return 0, today_date, 4

# In[20]:

def time_decorator(func): 
    def wrapper(**kwargs):
        start_time  = time.time() # 시작 시간 기록
        result = func(**kwargs) # 함수 실행
        end_time = time.time() # 종료 시간 기록
        print(f"{func.__name__} 함수 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

@time_decorator
def distribute_by_application(ship_id, op_index, section):

    # 데이터 로드
    df = get_dataframe_from_database('ecs_latest_data', ship_id = ship_id, op_index = op_index, section = section)
    # df = pd.read_csv(r"C:\Users\pc021\Desktop\dx_project\techross\health_data\data\latest\ecs_data.csv")
    # df = df[(df['SHIP_ID']==ship_id) & (df['OP_INDEX']==op_index) & (df['SECTION']==section)]

    optime = pd.read_csv(r"C:\Users\pc021\Desktop\dx_project\techross\health_data\data\latest\ecs_optime.csv")
    optime = optime[(optime['SHIP_ID']==ship_id) & (optime['OP_INDEX']==op_index) & (optime['SECTION']==section)]

    # optime = get_dataframe_from_database_optime(ship_id,op_index)

    df = df[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','TRO','ANU','RATE','CURRENT','VOLTAGE']]
    optime = optime[['SHIP_ID','OP_INDEX','OP_TYPE','START_TIME','END_TIME','RUNNING_TIME']]
    date_time = optime.iloc[0]['START_TIME']

    if optime.empty:
        logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=OP_TIME DATAFRAME 비어있습니다. | TYPE=PREPROCESSING | is_processed=False')
        print("optime DataFrame 비어있습니다.")
        return None, None

    # op_type 추출
    op_type = optime.iloc[0]['OP_TYPE']

    # Ballast 경우 진행
    if (op_type!=2) & (op_type!=4):
        # Data Merge
        sensor = pd.merge(optime, df, on=['SHIP_ID','OP_INDEX'], how='left')

        # 전처리 함수 적용
        try:
            original_data, data, indicator_data, text = apply_preprocessing_fuction(ship_id, op_index, section, sensor)
            if data is None:
                return original_data, None
            else:
                return original_data, data
            
        except ValueError as e :
            print(f'에러 발생: {e}. 다음 반복으로 넘어갑니다.')
            return sensor, None  
        
        # 에러 발생 시 다음 반복으로 넘어감\
        
        # 딕셔너리 텍스트 - 데이터 프레임 적용
        # dict_dataframe = pd.DataFrame([indicator_data])
        
        # 데이터 적재
        # load_database(dict_dataframe,'tc_data_preprocessing')
        
        # # 데이터 베이스 등록 데이터 추출
        # idx,reg,op_type = distribute_variables(ship_id, op_index, section)
        
        # 설명 텍스트 저장
        # with open(f'D:\\bwms\\prefix_json\\{reg}_{idx}.json', 'w', encoding='utf-8') as json_file:
        #     json.dump(text, json_file, ensure_ascii=False, indent=4)
            
        # 전처리 산출물 정제
        #original_data, data = select_data_variable(original_data, data)
        
        # 상태 변수 추가
    #    original_data['state'] = 'original'
    #    data['state'] ='preprocessing' 

        # 변수 결합
    #    concat_data = pd.concat([original_data, data])

    # 시각화 함수 적용
    #     if indicator_data['NOISE']:
    #         visual.plot_histograms_with_noise(concat_data, idx, reg, op_type)
    #         #visual.plot_histograms_with_noise(data,idx,reg,op_type,'after')
            
    #     if indicator_data['OPERATION']:
    #         visual.plot_bar_with_operation(original_data,idx,reg)
        
    #     if indicator_data['DUPLICATE']:
    #         visual.plot_pie_with_duplication(original_data,idx,reg)
        
    #     if indicator_data['MISSING']:
    #         visual.plot_double_bar_with_missing(original_data,idx,reg)
        
    #     # 실시간 데이터 적재
    #     data_col = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_INDEX']
    #     result_data = data[data_col]
    #     result_data['REG_DATE'] = reg
    #     # load_database(result_data,'tc_data_preprocessing_result')
    # else:
    #     print(f"{op_type}은 통과하지 않습니다.")
    else:
        logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=해당 OP_TIME은 DEBALLAST입니다. | TYPE=PREPROCESSING | is_processed=False')
        return None, None
    
    # 가동 시간을 측정을 위한 데코레이터 함수