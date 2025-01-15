# basic
import pandas as pd

# database
from sqlalchemy import create_engine


def get_dataframe_from_database(database, table_name, all=False, **kwargs):

    ship_id = kwargs.get('ship_id')
    op_index = kwargs.get('op_index')
    section = kwargs.get('section')
    
    # username = 'bwms_dba'
    # password = '!^admin1234^!'
    # host = 'signlab.iptime.org'  # 또는 서버의 IP 주소
    # port = 20002  # MariaDB의 기본 포트
        
    username = 'signlab'
    password = 'signlab123'
    host = '172.16.18.11'  # 또는 서버의 IP 주소
    port = 3306 # MariaDB의 기본 포트

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    

    if not all: 
        # SQL 쿼리
        query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}'AND `OP_INDEX` = '{op_index}' AND `SECTION` = '{section}';"
    else:
        # SQL 쿼리
        query = f"SELECT * FROM `{table_name}`"

    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df