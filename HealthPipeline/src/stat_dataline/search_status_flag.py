# basic
import pandas as pd
import json

# time
from datetime import datetime, timedelta

# database
from sqlalchemy import create_engine

# module
from stat_dataline.select_and_update_flag_status import get_and_update_flag_status 

def get_reference_dates_by_flag_status():
    """
    Extract reference dates from the `tc_flag_status` table where:
    - `IS_COMPLETE` is 1 (True)
    - `IS_PREPROCESSING` is not 0 (False)

    Returns:
        list: A list of reference dates (`REFERENCE_DT`) matching the conditions.
    """
   
    filtered_df = get_and_update_flag_status()

    # Convert the filtered reference dates to a list
    reference_dates = filtered_df['REFERENCE_DT'].tolist()

    print("/n[건전성 분석을 위한 날짜 추출]...")
    print(f"{reference_dates}")

    return reference_dates


def fetch_data_on_schedule(database, table_name, start_time, end_time):
    
    username = 'signlab'
    password = 'signlab123'
    host = '172.16.18.11'  # 또는 서버의 IP 주소
    port = 3306 # MariaDB의 기본 포트

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    
    query = f"""
    SELECT * FROM `{table_name}` 
    WHERE  `REG_DATE` BETWEEN '{start_time}' AND '{end_time}' AND 'FLAG' = 0;
    """
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("해당 데이터 프레임은 비어있습니다.")
        return None

    # FLAG 업데이트
    # update_query = f"""
    # UPDATE `{table_name}`
    # SET `FLAG` = 1
    # WHERE `REG_DATE` BETWEEN '{start_time}' AND '{end_time}' AND `FLAG` = 0;
    # """
    with engine.begin() as connection:
        connection.execute(query)

    # 결과 반환
    return df


def filter_by_flag_status():

    # 건전성 분석을 위한 날짜 추출
    reference_dates = get_reference_dates_by_flag_status()

    filtered_dataframes = [] # 필터링된 데이터프레임을 담을 리스트

    for date in reference_dates:

        # 기준 시각
        start_time  = date
        end_time = start_time + pd.Timedelta(days=1) # 하루 뒤

        filterd_data = fetch_data_on_schedule('signlab', 'tc_ecs_data_flag', start_time, end_time)
        filtered_dataframes.append(filterd_data) # 리스트에 데이터 추가 

    print("/n [건전성 분석 데이터 리턴..]")

    return pd.concat(filtered_dataframes, ignore_index=True)