# basic
import numpy as np
import pandas as pd


# module.healthchecker
import HealthPipeline.src.stat_healthchecker.deprecated_modules.rate_of_change_algorithms as rate_algorithms

# module.dataline
from stat_dataline.load_database import load_database

# set
import warnings
warnings.filterwarnings("ignore")


def refine_frames(data):
    # 변수 선택
    columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO']
    data = data[columns]
    
    return data



def apply_system_health_statistics_with_csu(data):
    """ 그룹 통계 함수 적용
    """

    # 데이터 그룹화
    group = data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
    
    # SCORE 변수 추가
    score = calculate_group_health_score(data,'CSU')
    
    group['HEALTH_SCORE'] = score
    
    # 데이터 인덱스 리셋
    group = group.reset_index()
    
    group = group[['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','HEALTH_SCORE']]

    # 데이터 적재
    #load_database()
    
    return group




def exceed_limit_line(data,col):
    limit={'CSU': 45,
        'STS' : 40}
    
    # 변수 선택
    val = data[col].max()
    
    if val > limit[col]:
        return 100
    else :
        return 0



def generate_health_score(data,col):
    
    #data.sort_values(by='DATA_INDEX',inplace=True)
    
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
    pre.columns=['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
                'CSU_Ratio','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    pre = pre.rename({'CSU_Ratio':'DIFF'},axis=1)
    
    # 범위 설정
    pre['HEALTH_RATIO']=pre['HEALTH_RATIO'].apply(lambda x : 100 if x >= 100 else x)
    
    
    return pre




def calculate_group_health_score(data,col):
    threshold={'CSU': 0.88,
        'STS' : 1.18}
    
    first = data.iloc[0][col]
    last = data.iloc[-1][col]

    trend_score=(abs(last - first) / threshold[col]) * 10

    health_score=data['HEALTH_RATIO'].max()
    
    total_score = health_score+trend_score
    
    limit_score = exceed_limit_line(data,col)
    
    return [ 100 if total_score + limit_score >= 100 else total_score + limit_score]




# Golas: 알고리즘 적용

def apply_system_health_algorithms_with_csu(data):
    """ CSU 건강도 알고리즘 적용 
    Args: 
     선박 이름, 오퍼레이션 번호, 섹션 번호
    
    Returns: 
     오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재

    """
    # 데이터 정제
    data = refine_frames(data)
    
    # 시스텀 건강도 데이터 셋 구축
    system_data = generate_health_score(data,'CSU')

    # 데이터 프레임 재 설정
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','CSU','DIFF',
                      'THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    system_data_condition = system_data[position_columns]
    
    # 그룹 적재 
    group = apply_system_health_statistics_with_csu(system_data_condition)

    return system_data_condition, group

