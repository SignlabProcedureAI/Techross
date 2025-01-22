# basic
import json
import pandas as pd
import time


# module
from PipelinePrep.src.prep.deprecated_modules.preprocessing import apply_preprocessing_fuction
from PipelinePrep.src.prep_visualizer.deprecated_modules import visualize as visual

# module.dataline
from prep_dataline  import get_dataframe_from_database
from prep_dataline import load_database
from stat_dataline import logger


def select_data_variable(original_data, data):
    original_data.rename({'ship_name':'SHIP_NAME'},axis=1)
    data = data.rename({'ship_name':'SHIP_NAME'},axis=1)
    
    return original_data,data

# def distribute_variables(ship_id, op_index, section):
    """ 데이터 추출 후 변수 적용
    """
    
    
#     # 데이터 베이스 등록 데이터 추출
#     # get_data = get_dataframe('tc_data_preprocessing',ship_id, op_index, section) 

#     # # 등록 시간 추출
#     # reg = get_data['REG_DATE'][0]
#     # reg = reg.strftime('%Y-%m-%d')

#     # # op_type 추출
#     # op_type = 'ba' if get_data['OP_TYPE'][0]!=2 else 'de'

#     # BOA 정의
#     #boa = 'before'
    
#     # return 

def find_folder(file_path):

    # 파일의 디렉토리 경로 추출
    directory = os.path.dirname(file_path)

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def process_preprocessed_data(indicator_data, preprocessed_data):
        
    # 딕셔너리 텍스트 - 데이터 프레임 적용
    dict_dataframe = pd.DataFrame([indicator_data])

    concat = pd.concat([dict_dataframe, preprocessed_data], axis=1)

    concat = concat[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','NOISE','MISSING','DUPLICATE','OPERATION','DATA_COUNT','PRE_COUNT','START_DATE','END_DATE','REG_DATE']]

    # 데이터 적재
    # load_database('signlab', 'tc_data_preprocessing', concat)
    load_database('ecs_test', 'test_tc_data_preprocessing_flag', concat)

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
    df = get_dataframe_from_database('ecs_data_new', ship_id = ship_id, op_index = op_index, section = section)
    optime = get_dataframe_from_database('ecs_optime_new', optime=True, ship_id = ship_id, op_index = op_index)

    df = df[['SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX','CSU','STS','FTS','FMU','TRO','ANU','RATE','CURRENT','VOLTAGE']]
    optime = optime[['SHIP_ID','OP_INDEX','OP_TYPE','START_TIME','END_TIME','RUNNING_TIME']] 
    date_time = optime.iloc[0]['START_TIME']

    if optime.empty:
        logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=op_time dataframe is empty| TYPE=preprocessing | IS_PROCESSED=False')
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
            original_data, data, indicator_data, data_preprocessed, text = apply_preprocessing_fuction(ship_id, op_index, section, sensor)
            if data is None:
                return original_data, None
            else:
                # 데이터 베이스 등록 데이터 추출
                # idx,reg,op_type = distribute_variables(ship_id, op_index, section)
                    
                # tc_data_preprocessing 적재
                process_preprocessed_data(indicator_data,data_preprocessed) 

                # 설명 텍스트 저장
                file_path = f'D:\\bwms_test\\{ship_id}\\{op_index}\\{ship_id}_{op_index}_{section}_file_ba.json'

                find_folder(file_path)

                with open(file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(text, json_file, ensure_ascii=False, indent=4)
                    
                # 전처리 산출물 정제
                original_data, data = select_data_variable(original_data, data)
                
                # 상태 변수 추가
                original_data['state'] = 'original'
                data['state'] ='preprocessing' 

                # 변수 결합
                concat_data = pd.concat([original_data, data])

                # 시각화 함수 적용
                if data_preprocessed['NOISE'].iloc[0] > 0:
                    visual.plot_histograms_with_noise(concat_data, ship_id, op_index, section, op_type)
                    
                if data_preprocessed['OPERATION'].iloc[0] > 0:
                    visual.plot_bar_with_operation(original_data,ship_id, op_index, section, op_type)
                
                if data_preprocessed['DUPLICATE'].iloc[0] > 0:
                    visual.plot_pie_with_duplication(original_data,ship_id, op_index, section, op_type)
                
                if data_preprocessed['MISSING'].iloc[0] > 0:
                    visual.plot_double_bar_with_missing(original_data,ship_id, op_index, section, op_type)
                
                # 실시간 데이터 적재
                data_col = ['SHIP_ID','OP_INDEX','SECTION','DATA_INDEX']
                result_data = data[data_col]
                # load_database('signlab', 'tc_data_preprocessing_result', result_data) 
                load_database('ecs_test', 'test_tc_data_preprocessing_result_flag', result_data)

                return original_data, data
            
        except ValueError as e :
            print(f'에러 발생: {e}. 다음 반복으로 넘어갑니다.')
            return sensor, None  
        
    else:
        logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=The op_type is deballast | TYPE=preprocessing | IS_PROCESSED=False')
        return None, None