#!/usr/bin/env python
# coding: utf-8

# In[1]:


# basic
import pandas as pd
import numpy as np
import time

# module.algorithms
import my_package.algorithms.csu_system_health_algorithms as csu_algorithms
import my_package.algorithms.sts_system_health_algorithms as sts_algorithms
import my_package.algorithms.fts_system_health_algorithms as fts_algorithms
import my_package.algorithms.fmu_system_health_algorithms as fmu_algorithms
import my_package.algorithms.faulty_algorithms as faulty_algorithms 
import my_package.algorithms.current_system_health_algorithms as current_algorithms

# module.data
from my_package.data.load_database import load_database

# In[2]:

def time_decorator(func): 
    def wrapper(*args, **kwargs):
        start_time  = time.time() # 시작 시간 기록
        result = func(*args, **kwargs) # 함수 실행
        end_time = time.time() # 종료 시간 기록
        print(f"{func.__name__} 함수 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper

# 데이터 로드
@time_decorator
def apply_system_health_algorithms_with_total(data, ship_id, op_index, section):
    
    # 1 .CSU 건강도 적용
    csu, csu_group = csu_algorithms.apply_system_health_algorithms_with_csu(data)

    # 2. STS 건강도 적용
    sts, sts_group = sts_algorithms.apply_system_health_algorithms_with_sts(data)

    # 3. TRO 불량 원인 적용
    tro, tro_group = faulty_algorithms.apply_fault_algorithms(data)
    
    # 4. FTS 건강도 적용
    fts, fts_group = fts_algorithms.apply_system_health_algorithms_with_fts(data)
  
    # 5. FMU 건강도 적용
    fmu, fmu_group = fmu_algorithms.apply_system_health_algorithms_with_fmu(data, ship_id)
    
    # 6. 전극 효율 적용
    current, current_group = current_algorithms.apply_system_health_algorithms_with_current(data)
    
    # 총 건강도 반환
    criteria = preprocess_system_health_algorithms_with_total(csu_group,sts_group,tro_group,fts_group,fmu_group,current_group)
    
    # criteria.to_csv(fr"C:\Users\pc021\Desktop\dx_project\techross\refrences\data_evaluation\health_data\{ship_id}_{op_index}_{section}.csv")

    # 그룹 적재
    load_database(criteria,'tc_ai_total_system_health_group')


# In[3]:


def generate_tro_health_score(tro): 
    
    # TRO LABEL 점수 합산
    label_sum = tro['STEEP_LABEL'] + tro['SLOWLY_LABEL'] +tro['OUT_OF_WATER_STEEP'] + tro['HUNTING'] + tro['TIME_OFFSET']
    
    # TRO LABEL 기준 당 * 30
    tro_health_score =label_sum[0] * 30
    
    # TRO 점수 수정
    tro_health_score = np.where(tro_health_score >= 100, 100, tro_health_score)
    
    return tro_health_score


# In[4]:


def round_up_sensor_values(data, columns):
    for column in columns:
        data[column] = np.round(data[column],2)

    return data

def preprocess_system_health_algorithms_with_total(csu,sts,tro,fts,fmu,current):
    
    #  CSU 데이터를 기준으로 설정
    criteria = csu[['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','HEALTH_SCORE']]
    # 변수 변경
    criteria.columns= ['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','CSU_HEALTH_SCORE']
    
    # STS
    sts = sts[['HEALTH_SCORE']]
    sts.columns=['STS_HEALTH_SCORE']
    criteria['STS_HEALTH_SCORE'] = sts['STS_HEALTH_SCORE']
    
    # FTS
    fts = fts[['HEALTH_SCORE']]
    fts.columns=['FTS_HEALTH_SCORE']    
    criteria['FTS_HEALTH_SCORE'] = fts['FTS_HEALTH_SCORE']
    
    # FMU
    fmu = fmu[['HEALTH_SCORE']]
    fmu.columns=['FMU_HEALTH_SCORE']    
    criteria['FMU_HEALTH_SCORE'] = fmu['FMU_HEALTH_SCORE']
    
    # CURRENT 
    current = current[['ELECTRODE_EFFICIENCY','START_TIME','END_TIME','RUNNING_TIME','OP_TYPE']]
    current.columns=['CURRENT_HEALTH_SCORE','START_TIME','END_TIME','RUNNING_TIME','OP_TYPE']
    criteria['CURRENT_HEALTH_SCORE'] = abs(current['CURRENT_HEALTH_SCORE'])
    
    # TRO 
    tro_health_score = generate_tro_health_score(tro)
    criteria['TRO_HEALTH_SCORE'] = tro_health_score
    
    # TOTAL
    criteria['TOTAL_HEALTH_SCORE'] = (criteria['CSU_HEALTH_SCORE'] + criteria['STS_HEALTH_SCORE'] + criteria['TRO_HEALTH_SCORE'] 
                                      + criteria['FTS_HEALTH_SCORE'] + criteria['FMU_HEALTH_SCORE'] + criteria['CURRENT_HEALTH_SCORE']) / 5
    
    # TOTAL 변수 선택
    data = criteria[['SHIP_ID','OP_INDEX','SECTION','CSU_HEALTH_SCORE','STS_HEALTH_SCORE','FTS_HEALTH_SCORE','FMU_HEALTH_SCORE','TRO_HEALTH_SCORE','CURRENT_HEALTH_SCORE'
          ,'TOTAL_HEALTH_SCORE']]
    
    data[['START_TIME','END_TIME','RUNNING_TIME','OP_TYPE']] = current[['START_TIME','END_TIME','RUNNING_TIME','OP_TYPE']]

    # 변수 반올림
    columns = ['CSU_HEALTH_SCORE','STS_HEALTH_SCORE','FTS_HEALTH_SCORE','FMU_HEALTH_SCORE','TRO_HEALTH_SCORE','CURRENT_HEALTH_SCORE'
          ,'TOTAL_HEALTH_SCORE']
    
    data = round_up_sensor_values(data,columns)

    return data

