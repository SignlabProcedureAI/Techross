# basic
import pandas as pd

# database
from sqlalchemy import create_engine


def get_ship_dataframe_from_database(database, table_name, ship_id):
    # 테이블 정보 설정
    #table_name='ecs_data'
    
    # 연결 정보 설정
    username = 'bwms_dba'  # 사용자 이름
    password = '!^admin1234^!'  # 비밀번호
    host = 'signlab.iptime.org'  # 서버 주소
    port = 20002  # 포트
    #database = 'ecs_test'  # 데이터베이스 이름

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    

    # SQL 쿼리
    query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}';"
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df

