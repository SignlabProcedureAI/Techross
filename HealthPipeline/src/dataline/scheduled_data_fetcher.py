#!/usr/bin/env python
# coding: utf-8

# In[1]:


# time
import time
from datetime import datetime, timedelta

# basic
import pandas as pd
import json

# db
from sqlalchemy import create_engine


# In[ ]:


def fetch_data_on_schedule(table_name):
    
    # 연결 정보 설정
    username = 'bwms_dba'  # 사용자 이름
    password = '!^admin1234^!'  # 비밀번호
    host = 'signlab.iptime.org'  # 서버 주소
    port = 20002  # 포트
    database = 'signlab'  # 데이터베이스 이름

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

