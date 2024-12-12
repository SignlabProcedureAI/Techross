# basic
import pandas as pd
import json

# time
from datetime import datetime, timedelta

# database
from sqlalchemy import create_engine

def fetch_data_on_schedule(database, table_name):
    
    username = 'signlab'
    password = ''
    host = '172.16.18.11'  # 또는 서버의 IP 주소
    port = 3306 # MariaDB의 기본 포트
 

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    
    # JSON 파일에서 불러오기
    with open("last_fetched.json", "r") as file:
        data = json.load(file)
        start_time = datetime.strptime(data["last_timestamp"], "%Y-%m-%d %H:%M:%S")

    # 현재 시간 저장 예시
    last_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"last_timestamp": last_timestamp}

        # JSON 파일에 저장
    with open("last_fetched.json", "w") as file:
        json.dump(data, file)

    query = f"""
    SELECT * FROM `{table_name}` 
    WHERE  `DATA_TIME` BETWEEN '{start_time}' AND '{last_timestamp}';
    """
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("해당 데이터 프레임은 비어있습니다.")
    # 결과 반환
    return df

