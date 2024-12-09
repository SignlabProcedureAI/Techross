#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sqlalchemy import create_engine, text

def delete_from_database(table_name):
    # MariaDB 연결 설정
    username = 'bwms_dba'
    password = '!^admin1234^!'
    host = 'signlab.iptime.org'  # 또는 서버의 IP 주소
    port = 20002  # MariaDB의 기본 포트
    database = 'ecs_dat1'
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

    # 데이터 삭제
    try:
        with engine.connect() as connection:
            # SQL 쿼리를 실행하여 특정 SHIP_ID에 해당하는 행 삭제
            query = text(f"DELETE FROM {table_name}")
            result = connection.execute(query)
            
            # 삭제된 행 수 확인
            print(f"Successfully deleted all records from {table_name}. Deleted rows: {result.rowcount}")

    except Exception as e:
        # 예외 처리
        print(f"Failed to delete data: {e}")

