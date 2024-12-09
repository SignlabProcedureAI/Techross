#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sqlalchemy import create_engine
import pandas as pd

def get_dataframe_from_database_fluid(table_name,ship_id,op_index,section):
    # 테이블 정보 설정
    
    # 연결 정보 설정
    username = 'signlab'  # 사용자 이름
    password = '!^admin1234^!'  # 비밀번호
    host = '192.168.0.248'  # 서버 주소
    port = 3306  # 포트
    database = 'signlab'  # 데이터베이스 이름

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    
    # SQL 쿼리
    query = f"SELECT * FROM `{table_name}` WHERE `SHIP_ID` = '{ship_id}'AND `OP_INDEX` = {op_index} AND `SECTION` = {section};"
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    
    return df

