#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sqlalchemy import create_engine
import pandas as pd

def get_ship_dataframe_from_database(table_name, ship_id, total=False):
    # 테이블 정보 설정
    
    # 연결 정보 설정
    username = 'bwms_dba'  # 사용자 이름
    password = '!^admin1234^!'  # 비밀번호
    host = 'signlab.iptime.org'  # 서버 주소
    port = 20002  # 포트
    database = 'ecs_dat1'  # 데이터베이스 이름

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    

    # SQL 쿼리
    query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}';"
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df

