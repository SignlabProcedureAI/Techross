# basic
import pandas as pd
import os
import pickle

# module.healthchecker
import models_healthchecker.rate_of_change_algorithms as rate_algorithms
import models_healthchecker.apply_hunting as apply_hunting
import models_healthchecker.apply_time_offset as apply_time_offset

# module.dataline
from models_dataline.load_database import load_database


def catorize_health_score(data):

    data['SUM'] = data['STEEP_LABEL']+data['SLOWLY_LABEL']+data['OUT_OF_WATER_STEEP']+data['HUNTING']+data['TIME_OFFSET']
    
    data['DEFECT_RISK_CATEGORY'] = 0

    data.loc[data['SUM']<=0, 'RISK'] = 'NORMAL'
    data.loc[(data['SUM']>0) & (data['SUM']<=2), 'RISK'] = 'WARNING'
    data.loc[(data['SUM']>2) & (data['SUM']<=6), 'RISK'] = 'RISK'
    data.loc[data['SUM']>6, 'RISK'] = 'DEFECT'

    data = data.drop(columns='SUM')

    return data

def give_tro_condition(data):
    # TRO 조건 부여
    data.loc[data['TRO'] >= 8,'STEEP_LABEL'] = 0
    
    return data
    
  

def give_tro_out_of_water_condition(data):
    data['OUT_OF_WATER_STEEP'] = 0
    # 조건 부여
    data.loc[(data['TRO'] <= 1) & (data['STEEP_LABEL']==1) ,'OUT_OF_WATER_STEEP'] = 1
    data.loc[(data['TRO'] <= 1) & (data['STEEP_LABEL']==1) ,'STEEP_LABEL'] = 0
    
    return data


def refine_frames(data):
    # 변수 선택
    columns=['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','START_TIME','END_TIME','RUNNING_TIME']
    data=data[columns]
    
    return data


def apply_automation_labeling(data):
    """ 자동화 라벨링 알고리즘을 적용 함수
    
    """
    # Steep Label 함수 적용
    data['steep_label'] = data.apply(lambda x : rate_algorithms.classify_decline_steep_label(x['TRO_Ratio'],x['pred_Ratio']),axis=1)
    # Slowly Label 함수 적용
    #data['slowly_label'] = data.apply(lambda x : rate_algorithms.classify_decline_slowly_label(x['TRO_NEG_COUNT']),axis=1)
    
    # 이전 값을 나타내는 열 추가
    data['Previous_TRO_NEG_COUNT'] = data['TRO_NEG_COUNT'].shift(1)
    
    # shif(1)에 의한 결측치 제거
    data.dropna(inplace=True)
    data['slowly_label'] = data.apply(lambda x : rate_algorithms.classify_decline_slowly_label(x['Previous_TRO_NEG_COUNT'],x['TRO_NEG_COUNT']),axis=1)
    
    data = data.drop(columns='Previous_TRO_NEG_COUNT')
    
    return data


def apply_fault_label_statistics(data,count):
    """ 그룹 통계 함수 적용
    """
    
    
    data['DATA_TIME'] = pd.to_datetime(data['DATA_TIME'])
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # 시간 추출
    start_date = data.iloc[0,24]
    end_date = data.iloc[0,25]
    running_time = data.iloc[0,26]
    op_type = data.iloc[0,3]

    
    # 데이터 그룹화
    group=data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg({'CSU':'mean','STS':'mean','FTS':'mean','FMU':'mean','CURRENT':
                                                             'mean','TRO':['min','mean','max'],'TRO_DIFF':['min','mean','max'],'TRO_PRED':'mean','PEAK_VALLEY_INDICES':'sum','CROSS_CORRELATION':'mean','RE_CROSS_CORRELATION':'mean',
                                                       'PRED_DIFF':'mean','TRO_NEG_COUNT':'max','STEEP_LABEL':'sum','SLOWLY_LABEL':'sum','OUT_OF_WATER_STEEP':'sum','HUNTING':'sum','TIME_OFFSET':'sum'})
    
    # 다중 인덱스된 컬럼을 단일 레벨로 평탄화
    group.columns = ['_'.join(col) for col in group.columns]
    group.columns = ['CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX','TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','TRO_PRED','PEAK_VALLEY_INDICES_SUM','CROSS_CORRELATION','RE_CROSS_CORRELATION','PRED_DIFF',
                     'TRO_NEG_COUNT','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET']


    # 데이터 인덱스 리셋
    group = group.reset_index()
    
    group['START_TIME'] = start_date
    group['END_TIME'] = end_date
    group['RUNNING_TIME'] = running_time 
    group['OP_TYPE'] = op_type 
    group['RE_CROSS_CORRELATION_COUNT'] = count
    
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX','TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','PEAK_VALLEY_INDICES_SUM','CROSS_CORRELATION','RE_CROSS_CORRELATION','PRED_DIFF','RE_CROSS_CORRELATION_COUNT',
                     'TRO_NEG_COUNT','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET','START_TIME','END_TIME','RUNNING_TIME']]
    
    # 학습 데이터 적재
    load_database('ecs_test','tc_ai_fault_group_v1.1.0', '200', group)

    # 모델 로드

    # 현재 파일의 경로를 기준으로 model 폴더 내 tro_model 경로 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tro_model_relative_path = os.path.join(current_dir,".." , "models_model","tro_group_model_v2.0.0")
    # 상대 경로를 절대 경로로 변환
    tro_model_path = os.path.abspath(tro_model_relative_path)

    model = load_model_from_pickle(tro_model_path)

    # # 변수 선택 및 예측
    X = group[['CSU', 'STS', 'FTS', 'FMU', 'CURRENT', 'TRO_MIN', 'TRO_MEAN', 'TRO_MAX',
       'TRO_DIFF_MIN', 'TRO_DIFF_MEAN', 'TRO_DIFF_MAX', 'TRO_NEG_COUNT',
       'PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION',
       'RE_CROSS_CORRELATION_COUNT']]
    
    pred =  model.predict(X)

    group['PRED'] = pred

    # 웹 표출을 위한 변수 정제
    group = group[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET','PRED','START_TIME','END_TIME','RUNNING_TIME']]
    group = catorize_health_score(group)

    # 데이터 적재
    load_database('signlab','tc_ai_fault_model_group', 'release', group)
    # load_database('ecs_test','tc_ai_fault_model_group_flag', '200', group)
    
    return group



def model_predict(data):
    
    # 데이터 컬럼 선택
    columns = ['CSU','STS','FTS','FMU','CURRENT']
    
    # TRO 모델 로드
    dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(dir, '../'))
    tro_model_path = os.path.join(parent_dir,'models_model', 'tro_model')

    tro_model = rate_algorithms.load_pickle(tro_model_path)

    # 독립변수와 종속변수 추출
    independent_variable = data[columns]
    dependent_variable = data['TRO']

    # TRO 예측
    predicted_tro = tro_model.predict(independent_variable)

    # 예측 값 변수 생성
    data['pred'] = predicted_tro

    return data



def label_data_points(data):

     # 변화율 변수 생성 및 정제
    data = rate_algorithms.calculating_rate_change(data,'TRO')
    data = rate_algorithms.calculating_rate_change(data,'pred')
    data = data.dropna()

    # 데이터 프레임 재 설정
    position_columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','TRO_Ratio','pred','pred_Ratio','START_TIME','END_TIME','RUNNING_TIME']
    data=data[position_columns]
                                                    
    # 알고리즘을 적용하여 TRO_NEG_COUNT 생성
    data = rate_algorithms.generate_tro_neg_count(data) 
    
    # 자동화 라벨링 적용
    data = apply_automation_labeling(data)
    
    # 컬럼 변경
    data.columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','TRO_DIFF','TRO_PRED','PRED_DIFF','START_TIME','END_TIME','RUNNING_TIME','TRO_NEG_COUNT',
                'STEEP_LABEL','SLOWLY_LABEL']
    
    # 조건 추가
    data = give_tro_condition(data)
    
    # out of water steep 
    data = give_tro_out_of_water_condition(data)
    
    # Hunting 라벨 
    data = apply_hunting.label_hunting_multiple_of_two(data)
    
    # Time Offset 라벨 
    data, count = apply_time_offset.classify_time_offset_label(data)

    # 16. 변수명 변경
    data = data.rename({'Hunting':'HUNTING'},axis=1)

    return data, count



def apply_fault_algorithms(data):
    """ TRO 알고리즘 적용
    Args : 선박 이름, 오퍼레이션 번호, 섹션 번호
    
    Returns: 오퍼레이션 실시간 데이터 자동적재, 오퍼레이션 그룹 자동적재
    """

    # 1. 데이터 정제
    data = refine_frames(data)

    # 2. TRO 예측 함수 적용
    data = model_predict(data)
 
     # 3. 자동화 라벨링 함수 적용
    data, count = label_data_points(data)

    # 3. 변수 선택
    data = data[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','TRO_DIFF','TRO_PRED','PRED_DIFF',
                    'PEAK_VALLEY_INDICES','RE_CROSS_CORRELATION','CROSS_CORRELATION',
            'TRO_NEG_COUNT','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET','START_TIME','END_TIME','RUNNING_TIME']]

    # 4. 자동 적재
    # load_database()

    # 5. 그룹 적재 
    group = apply_fault_label_statistics(data, count)

    return data, group


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