# basic
import pandas as pd

# module
from stat_dataline import DatabaseEngine

def get_dataframe_from_database(table_name, all: bool = False, optime: bool = False, **kwargs):

    ship_id = kwargs.get('ship_id')
    op_index = kwargs.get('op_index')
    section = kwargs.get('section')
    
    # username = 'bwms_dba'
    # password = '!^admin1234^!'
    # host = 'signlab.iptime.org'  # 또는 서버의 IP 주소
    # port = 20002  # MariaDB의 기본 포트
    
    # DatabaseEngine 객체 생성
    # db_engine = DatabaseEngine(
    #     username='signlab',
    #     password='signlab123',
    #     host='172.16.18.11',
    #     port=3306,
    #     database='signlab'
    # )

    db_engine = DatabaseEngine(
        username='bwms_dba',
        password='!^admin1234^!',
        host='signlab.iptime.org',
        port=20002,
        database='ecs_dat1'
    )

    engine = db_engine.engine  # SQLAlchemy 엔진 접근

    if not all and not optime: 
        # SQL 쿼리
        query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}'AND `OP_INDEX` = '{op_index}' AND `SECTION` = '{section}';"
    elif not all and optime:
        query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}'AND `OP_INDEX` = {op_index} ;"
    else:
        # SQL 쿼리
        query = f"SELECT * FROM `{table_name}`"

    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df


