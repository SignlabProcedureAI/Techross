#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine
import io

def load_database(data,table_name):

    # MariaDB 연결을 설정합니다.
    # 'username', 'password', 'host', 'port', 'database'를 실제 값으로 대체하세요.
    username = 'signlab'
    password = '!^admin1234^!'
    host = '192.168.0.248'  # 또는 서버의 IP 주소
    port = 3306  # MariaDB의 기본 포트
    database = 'signlab'
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

    # 데이터 프레임을 MariaDB에 적재합니다.
    # 'your_table_name'을 실제 테이블 이름으로 대체하세요.

    try:
        # DataFrame 'data'가 비어있는지 확인
        if not data.empty:
            data.to_sql(table_name, con=engine, if_exists='append', index=False)
            print(f'DataFrame has been successfully loaded into {table_name} table in {database} database.')
        else:
            # 빈 데이터 프레임인 경우
            print("The DataFrame is empty. No data was loaded into the table.")
            
    except Exception as e:
        # 예외 발생 
        print(f"Failed to load data:{e}")