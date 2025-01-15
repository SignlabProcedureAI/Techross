# basic
import pandas as pd
import os

# module.healthchecker
import HealthPipeline.src.stat_healthchecker.deprecated_modules.rate_of_change_algorithms as rate_algorithms
import hunting_processor as StatHungting
import stat_healthchecker.apply_time_offset as apply_time_offset

# module.dataline
from stat_dataline.load_database import load_database


def limit_date_time_label(data):
    data.iloc[:21,5] = 0
    data.iloc[-10:,7] = 0

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
# Goals:정제 함수

def refine_frames(data):
    # 변수 선택
    columns = ['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','CURRENT','TRO','START_TIME','END_TIME','RUNNING_TIME']
    data = data[columns]
    
    return data




def apply_automation_labeling(data):
    """ 자동화 라벨링 알고리즘을 적용
    """
    
    # Steep Label 함수 적용
    data['steep_label'] = data.apply(lambda x : rate_algorithms.classify_decline_steep_label(x['TRO_Ratio'],x['pred_Ratio']),axis=1)
    # Slowly Label 함수 적용
    
    # 이전 값을 나타내는 열 추가
    data['Previous_TRO_NEG_COUNT'] = data['TRO_NEG_COUNT'].shift(1)
    
    # shif(1)에 의한 결측치 제거
    data.dropna(inplace=True)
    data['slowly_label'] = data.apply(lambda x : rate_algorithms.classify_decline_slowly_label(x['Previous_TRO_NEG_COUNT'],x['TRO_NEG_COUNT']),axis=1)
    
    data = data.drop(columns='Previous_TRO_NEG_COUNT')
    
    return data


def apply_fault_label_statistics(data):
    """그룹 통계 함수 적용
    
    Args:
     
    
    Return: 
     
    """

    data['DATA_TIME'] = pd.to_datetime(data['DATA_TIME'])
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # 시간 추출
    start_date = data.iloc[0,10]
    end_date = data.iloc[0,11]
    running_time = data.iloc[0,12]
    op_type = data.iloc[0,13]

    # 데이터 그룹화
    group = data.groupby(['SHIP_ID','OP_INDEX','SECTION']).agg({'STEEP_LABEL':'sum','SLOWLY_LABEL':'sum','OUT_OF_WATER_STEEP':'sum','HUNTING':'sum','TIME_OFFSET':'sum'})

    # 데이터 인덱스 리셋
    group = group.reset_index()

    group['START_TIME'] = start_date
    group['END_TIME'] = end_date
    group['RUNNING_TIME'] = running_time 
    group['OP_TYPE'] = op_type 

    # 데이터 변수 재 설정
    group = group[['SHIP_ID','OP_INDEX','SECTION','RUNNING_TIME','OP_TYPE','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET','START_TIME','END_TIME']]

    # 데이터 적재
    load_database('signlab', 'tc_ai_fault_group', group)
    # load_database('ecs_test', 'tc_ai_fault_group_flag', group)

    return group


def model_predict(data):
    
    # 데이터 컬럼 선택
    columns = ['CSU','STS','FTS','FMU','CURRENT']
    
    # TRO 모델 로드
    dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(dir, '../'))
    tro_model_path = os.path.join(parent_dir,'stat_model', "tro_model")

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
    data = StatHungting.label_hunting_multiple_of_two(data)
    
    # Time Offset 라벨 
    data = apply_time_offset.classify_time_offset_label(data)

    # 16. 변수명 변경 (삭제)
    data = data.rename({'Hunting':'HUNTING'},axis=1)

    return data

def apply_fault_algorithms(data):
    """ TRO 건강도 알고리즘 적용 
    Args: 
     선박 이름, 오퍼레이션 번호, 섹션 번호
    
    Returns: 
     오퍼레이션 실시간 건강도 데이터 자동적재, 오퍼레이션 건강도 그룹 자동적재

    """

    # 1. 데이터 정제
    data = refine_frames(data)
    
    # 2. TRO 예측 함수 적용
    data = model_predict(data)

    # 3. 자동화 라벨링 함수 적용
    data = label_data_points(data)
    
    # 4. 변수 선택
    data = data[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET','START_TIME','END_TIME','RUNNING_TIME','OP_TYPE']]
    
    # 5. 제한 조건 부여
    data = limit_date_time_label(data)

    # 6. 그룹 적재 
    group = apply_fault_label_statistics(data)

    # 7. 최종 센서 데이터 변수 선택
    sensor_data = data[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','STEEP_LABEL','SLOWLY_LABEL','OUT_OF_WATER_STEEP','HUNTING','TIME_OFFSET']]

    # 8. 자동 적재
    # load_database('signlab', 'tc_ai_fault_label', sensor_data)

    return sensor_data, group