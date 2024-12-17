# basic
import pandas as pd

# database
from sqlalchemy import create_engine


def get_dataframe_from_database_optime(database, table_name, ship_id, op_index):
    # 테이블 정보 설정

    # 연결 정보 설정

    username = 'bwms_dba'
    password = '!^admin1234^!'
    host = 'signlab.iptime.org'  # 또는 서버의 IP 주소
    port = 20002  # MariaDB의 기본 포트
 

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    
    # SQL 쿼리
    query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}'AND `OP_INDEX` = {op_index} ;"
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df