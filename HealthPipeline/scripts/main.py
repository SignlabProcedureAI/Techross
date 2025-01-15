import sys
import os

# 경로 설정: 스크립트 경로에서 상위 디렉토리로 이동한 후 src 경로 추가
health_data_path = os.path.abspath(os.path.join('..', 'src'))
health_learning_data_path = os.path.abspath(os.path.join(os.getcwd(), "../../HealthModelPipeline/dataflow/src"))
preprocessing_path = os.path.abspath(os.path.join(os.getcwd(), "../../PipelinePrep/src"))

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

# basic
import pandas as pd

# time
import schedule
import time

# module.healthchecker
from stat_healthchecker.total_system_health_algorithm import apply_system_health_algorithms_with_total
from models_healthchecker.total_system_health_learning_algorithm import apply_system_health_learning_algorithms_with_total
from prep.load_processing import distribute_by_application

# module.dataline
from stat_dataline.logger_confg import logger
from stat_dataline.search_status_flag import DataPipelineManager


def get_latest_date_on_schedule():

    pipeline = DataPipelineManager()
    fetched_data = pipeline.filter_by_flag_status()
    # fetched_data = filter_by_flag_status()
    
    # 해당 추출 데이터 그룹화
    grouped_data = fetched_data.groupby(['SHIP_ID','OP_INDEX','SECTION']).count() 
    
    # 그룹 후 인덱스 추출 
    grouped_index = grouped_data.index

    return grouped_index, fetched_data



def schedule_health_assessment():
    
    grouped_index, fetched_data = get_latest_date_on_schedule()
    
    for index in grouped_index:

        # 해당 오퍼레이션 선박, 인덱스, 섹션 추출
        ship_id =  index[0]
        op_index = index[1]
        section =  index[2] 
        
        # 데이터 처리를 위한 갯수 조건을 만족하는지 판단 
        selected_df = fetched_data[(fetched_data['SHIP_ID']==ship_id) & (fetched_data['OP_INDEX']==op_index) & (fetched_data['SECTION']==section)]

        # 해당 오퍼레이션 데이터 길이 추출
        data_len = len(selected_df)

        # 해당 오퍼레이션 시작 시간 추출
        date_time = selected_df.iloc[0]['DATA_TIME']
       
        print(f'SHIP_ID : {ship_id} / OP_INDEX : {op_index} / SECTION : {section} -  데이터 선택 ({data_len})')
        
        if (data_len>=160) :
            
            print(f'SHIP_ID : {ship_id} / OP_INDEX : {op_index} / SECTION : {section} -  조건 통과')      
            
            try:
                sensor, preprocessed = distribute_by_application(ship_id=ship_id, op_index=op_index, section=section)
                if sensor is None and preprocessed is None:
                    print("선박 데이터 프레임이 존재하지 않습니다.")
                    continue       

                elif preprocessed is not None:
                    logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=The results were derived from the model and statistics package | TYPE=all | IS_PROCESSED=True')
                    print("전처리 후 학습 데이터 프레임이 존재합니다.")
                    apply_system_health_algorithms_with_total(sensor, ship_id, op_index, section)
                    apply_system_health_learning_algorithms_with_total(data=preprocessed, ship_id=ship_id, op_index=op_index, section=section)
                else:
                    logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time}  | LOG_ENTRY=After preprocessing, the model data frame does not exist, so only the statistical algorithm proceeds alone | TYPE=stats | IS_PROCESSED=True')
                    print("전처리 후 모델 데이터 프레임이 존재하지 않아 통계 알고리즘 단독 진행합니다.")
                    apply_system_health_algorithms_with_total(data=sensor, ship_id=ship_id, op_index=op_index, section=section)
                
            except ValueError as e :
                logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY={e} | TYPE=exceptional_handling | IS_PROCESSED=False')
                print(f'에러 발생: {e}. 다음 반복으로 넘어갑니다.')
                continue  # 에러 발생 시 다음 반복으로 넘어감\

            except KeyError as e :
                logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY={e} | TYPE=exceptional_handling | IS_PROCESSED=False')
                print(f'에러 발생: {e}. 다음 반복으로 넘어갑니다.')

            except TypeError as e :
                logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY={e} | TYPE=exceptional_handling | IS_PROCESSED=False')
                print(f'에러 발생: {e}. 다음 반복으로 넘어갑니다.')
                continue  # 에러 발생 시 다음 반복으로 넘어감
                
            except IndexError as e :
                print(f'에러 발생: {e}. 다음 반복으로 넘어갑니다.')
                logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY={e} | TYPE=exceptional_handling | IS_PROCESSED=False')
                continue  # 에러 발생 시 다음 반복으로 넘어감 
        else:
            logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=The data length is {data_len} and does not satisfy the condition | TYPE=data_length_limit | IS_PROCESSED=False')



# 스케줄 설정: 3일에 한 번씩 데이터 가져오기
schedule.every(1).days.at("15:02").do(schedule_health_assessment)
# schedule.every(5).minutes.do(schedule_health_assessment)

# 스케줄 지속 실행
while True:
    print("스케줄 시작")
    schedule.run_pending()
    time.sleep(1)

