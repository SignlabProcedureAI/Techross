# basic
import numpy as np
import time
import pandas as pd

# module.healthchecker
from csu_system_health import SimpleCsuSystemHealth 
from sts_system_health import SimpleStsSystemHealth 
from fts_system_health import SimpleFtsSystemHealth 
from fmu_system_health import SimpleFmuSystemHealth 
from tro_fault_detector import TROFaultAlgorithm
from current_system_health import SimpleCurrentSystemHealth 

# module.dataline
from stat_dataline.load_database import load_database

def time_decorator(func):
    """
    함수의 실행 시간을 측정하는 데코레이터.

    Args:
        func (callable): 데코레이터가 적용될 함수.

    Returns:
        callable: 실행 시간을 측정하고 결과를 반환하는 래퍼 함수
    """ 
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
    """
    시스템 건강 알고리즘을 순차적으로 적용하는 함수.

    Args:
        data (pd.DataFrame): 입력 데이터 프레임.
        ship_id (str): 선박 ID.
        op_index (int): 작업 인덱스.
        section (str): 섹션 정보.
    """
    # 알고리즘 클래스와 메서드 관리
    algorithms = [
        ("csu", SimpleCsuSystemHealth, "apply_system_health_algorithms_with_csu"),
        ("sts", SimpleStsSystemHealth, "apply_system_health_algorithms_with_sts"),
        ("tro", TROFaultAlgorithm, "apply_tro_fault_detector"),
        ("fts", SimpleFtsSystemHealth, "apply_system_health_algorithms_with_fts"),
        ("fmu", SimpleFmuSystemHealth, "apply_system_health_algorithms_with_fmu"),
        ("current", SimpleCurrentSystemHealth, "apply_system_health_algorithms_with_current")
    ]

    # 결과 저장용 딕셔너리
    results = {}
    group_results = {}

    # 각 알고리즘 실행 및 결과 저정
    for key, cls, method in algorithms:
        instance = cls(data)
        method_func = getattr(instance, method)
        result, group = method_func(status=False)
        result[key] = result
        group_results[key] = group
    
    # 총 건강도 기준 계산
    health_score_df =  preprocess_system_health_algorithms_with_total(
        *group_results.values()
    )
    # 결과 적재
    load_database('signlab', 'tc_ai_total_system_health_group', health_score_df)

def generate_tro_health_score(tro_df: pd.DataFrame) -> float: 
    """
    TRO 데이터를 기반으로 건강 점수를 생성하는 함수.

    Args:
        tro (pd.DataFrame): TRO 데이터프레임. 
            포함된 열:
            - 'STEEP_LABEL': 급격한 변화를 나타내는 레이블.
            - 'SLOWLY_LABEL': 느린 변화를 나타내는 레이블.
            - 'OUT_OF_WATER_STEEP': 수면 위 급격한 변화를 나타내는 레이블.
            - 'HUNTING': 헌팅 현상을 나타내는 레이블.
            - 'TIME_OFFSET': 시간 차이를 나타내는 레이블.

    Returns:
        int: TRO 건강 점수 (최대 100).
    """
     
    label_sum = tro_df['STEEP_LABEL'] + tro_df['SLOWLY_LABEL'] +tro_df['OUT_OF_WATER_STEEP'] + tro_df['HUNTING'] + tro_df['TIME_OFFSET']
    tro_health_score =label_sum[0] * 30
    tro_health_score = np.where(tro_health_score >= 100, 100, tro_health_score)
    return tro_health_score



def round_up_sensor_values(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    지정된 열의 값을 소수점 두 번째 자리로 반올림하는 함수.

    Args:
        data (pd.DataFrame): 처리할 데이터프레임.
        columns (list of str): 반올림할 열의 이름 리스트.

    Returns:
        pd.DataFrame: 지정된 열의 값이 반올림된 데이터프레임.
    """
    for column in columns:
        data[column] = np.round(data[column],2)

    return data

def preprocess_system_health_algorithms_with_total(csu: pd.DataFrame, sts: pd.DataFrame, tro: pd.DataFrame, fts: pd.DataFrame, fmu: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """"
    시스템 건강 알고리즘 데이터를 통합하여 최종 데이터프레임 생성.

    Args:
        csu (pd.DataFrame): CSU 건강 데이터.
        sts (pd.DataFrame): STS 건강 데이터.
        tro (pd.DataFrame): TRO 건강 데이터.
        fts (pd.DataFrame): FTS 건강 데이터.
        fmu (pd.DataFrame): FMU 건강 데이터.
        current (pd.DataFrame): CURRENT 건강 데이터.

    Returns:
        pd.DataFrame: 통합된 시스템 건강 데이터프레임.
    """
    
     # CSU 데이터 초기 설정
    health_score_df = csu[['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_INDEX', 'HEALTH_SCORE']].rename(
        columns={'HEALTH_SCORE': 'CSU_HEALTH_SCORE'}
    )
    
    # HEALTH_SCORE 통합
    health_data = {
        'STS_HEALTH_SCORE': sts,
        'FTS_HEALTH_SCORE': fts,
        'FMU_HEALTH_SCORE': fmu
    }
    for key, df in health_data.items():
            health_score_df[key] = df[['HEALTH_SCORE']].squeeze()

    # CURRENT 데이터 처리
    current = current[['ELECTRODE_EFFICIENCY', 'START_TIME', 'END_TIME', 'RUNNING_TIME', 'OP_TYPE']].rename(
        columns={'ELECTRODE_EFFICIENCY': 'CURRENT_HEALTH_SCORE'}
    )
    health_score_df['CURRENT_HEALTH_SCORE'] = abs(current['CURRENT_HEALTH_SCORE'])
        

    ## TRO 데이터 처리
    health_score_df['TRO_HEALTH_SCORE'] = generate_tro_health_score(tro)

    # TOTAL HEALTH SCORE 계산
    health_score_df['TOTAL_HEALTH_SCORE'] = (
        health_score_df[['CSU_HEALTH_SCORE', 'STS_HEALTH_SCORE', 'TRO_HEALTH_SCORE',
                  'FTS_HEALTH_SCORE', 'FMU_HEALTH_SCORE', 'CURRENT_HEALTH_SCORE']]
        .sum(axis=1) / 5
    )
    
   # 필요한 컬럼 선택 및 CURRENT의 시간 정보 추가
    data = health_score_df[[
        'SHIP_ID', 'OP_INDEX', 'SECTION', 'CSU_HEALTH_SCORE', 'STS_HEALTH_SCORE', 
        'FTS_HEALTH_SCORE', 'FMU_HEALTH_SCORE', 'TRO_HEALTH_SCORE', 
        'CURRENT_HEALTH_SCORE', 'TOTAL_HEALTH_SCORE'
    ]]
    data[['START_TIME', 'END_TIME', 'RUNNING_TIME', 'OP_TYPE']] = current[['START_TIME', 'END_TIME', 'RUNNING_TIME', 'OP_TYPE']]

    # HEALTH_SCORE 열 반올림
    score_columns = [
        'CSU_HEALTH_SCORE', 'STS_HEALTH_SCORE', 'FTS_HEALTH_SCORE',
        'FMU_HEALTH_SCORE', 'TRO_HEALTH_SCORE', 'CURRENT_HEALTH_SCORE',
        'TOTAL_HEALTH_SCORE'
    ]
    data = round_up_sensor_values(data, score_columns)

    return data

