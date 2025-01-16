# basic
import pandas as pd
import numpy as np
import time


# module.healthchecker
from csu_system_health import ModelCsuSystemHealth
from sts_system_health import ModelStsSystemHealth
import models_healthchecker.fts_system_health_algorithms as fts_algorithms
import models_healthchecker.fmu_system_health_algorithms as fmu_algorithms
from tro_fault_detector import TROFaultAlgorithm
import models_healthchecker.current_system_health_algorithms as current_algorithms

# module.dataline
from models_dataline.load_database import load_database



def time_decorator(func): 
    def wrapper(*args, **kwargs):
        start_time  = time.time() # 시작 시간 기록
        result = func(*args, **kwargs) # 함수 실행
        end_time = time.time() # 종료 시간 기록
        print(f"{func.__name__} 함수 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper



@time_decorator
def apply_system_health_learning_algorithms_with_total(data, ship_id, op_index, section):
    
    csu_instance = ModelCsuSystemHealth(data)
    csu_instance.apply_system_health_algorithms_with_csu(status=True)

    sts_instance = ModelStsSystemHealth(data)
    sts_instance.apply_system_health_algorithms_with_sts(status=True)

    fault_detector = TROFaultAlgorithm(data)
    fault_detector.apply_tro_fault_detector(status=True)
    
    # 4. FTS 건강도 적용
    fts, fts_group = fts_algorithms.apply_system_health_algorithms_with_fts(data)
  
    # 5. FMU 건강도 적용
    fmu, fmu_group = fmu_algorithms.apply_system_health_algorithms_with_fmu(data, ship_id)
    
    # 6. 전극 효율 적용
    current, current_group = current_algorithms.apply_system_health_algorithms_with_current(data)
    
    # 총 건강도 반환
    # criteria = preprocess_system_health_algorithms_with_total(csu_group,sts_group,tro_group,fts_group,fmu_group,current_group)
    
    # 그룹 적재
    # load_database('ecs_test', 'tc_ai_total_system_health_group_v0.0.0', criteria)

    #print(f"학습 데이터의 총 건강도 데이터 반환 : {criteria}")



def generate_tro_health_score(tro): 
    
    # TRO LABEL 점수 합산
    label_sum = tro['STEEP_LABEL'] + tro['SLOWLY_LABEL'] +tro['OUT_OF_WATER_STEEP'] + tro['HUNTING'] + tro['TIME_OFFSET']
    
    # TRO LABEL 기준 당 * 30
    tro_health_score =label_sum[0] * 30
    
    tro_health_score = np.where(tro_health_score >= 100, 100, tro_health_score)
    
    return tro_health_score


def preprocess_system_health_algorithms_with_total(csu,sts,tro,fts,fmu,current):
    
    #  CSU 데이터를 기준으로 설정
    criteria = csu[['SHIP_ID','OP_INDEX','SECTION','HEALTH_SCORE']]
    # 변수 변경
    criteria.columns= ['SHIP_ID','OP_INDEX','SECTION','CSU_HEALTH_SCORE']
    
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

    return data

