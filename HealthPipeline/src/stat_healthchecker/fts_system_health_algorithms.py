# basic
import numpy as np
import pandas as pd
import warnings

# module.healthcheckerhealthchecker
import stat_healthchecker.rate_of_change_algorithms as rate_algorithms

# module.dataline
from stat_dataline.load_database import load_database


# set 
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',2400)


def refine_frames(data):
    """ 데이터 정제 함수
    """
    # 1. 컬럼 설정
    columns=['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO']
    
     # 2.변수 선택
    data=data[columns]
    
    return data



def apply_system_health_statistics_with_fts(data):
    
    """ 그룹 통계 함수
    """
    # 1.데이터 그룹화
    group=data.groupby(['SHIP_ID','OP_INDEX','SECTION']).mean()
    
    # 2. Score 변수 추가
    score = calculate_group_health_score(data,'FTS')
    group['HEALTH_SCORE'] = score
    
    # 3. 데이터 인덱스 리셋
    group=group.reset_index()
    
    # 4, 변수 선택
    group=group[['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FTS','HEALTH_SCORE']]
    
    # 5. 데이터 적재
    #load_database()
    
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
    pre.columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO',
                'FTS_Ratio','THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
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
    
    total_score = health_score+trend_score
    
    limit_score = exceed_limit_line(data,col)
    
    
    return [ 100 if total_score + limit_score >= 100 else total_score + limit_score]



def apply_system_health_algorithms_with_fts(data):
    """ FTS 건강도 알고리즘 적용 
    Args: 
     선박 이름, 오퍼레이션 번호, 섹션 번호
    
    Returns: 
     오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재

    """
    
    # 1. 데이터 정제
    data = refine_frames(data)
    
    # 2. 시스텀 건강도 데이터 셋 구축
    system_data = generate_health_score(data,'FTS')

    # 3. 데이터 프레임 재 설정
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX','FTS','DIFF',
                      'THRESHOLD','HEALTH_RATIO','HEALTH_TREND']
    
    # 4. 변수 선택
    system_data_condition = system_data[position_columns]                                              
    
    # 5. 그룹 적재 
    group = apply_system_health_statistics_with_fts(system_data_condition)

    return system_data_condition, group

