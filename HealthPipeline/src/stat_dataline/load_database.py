# basic
import pandas as pd

# database
from sqlalchemy import create_engine
from sqlalchemy.types import VARCHAR, Integer, Float, Boolean, DateTime

def load_database(database, table_name, data):

    """
    pandas DataFrame을 MariaDB 테이블에 적재합니다.

    이 함수는 MariaDB 데이터베이스에 연결한 뒤, DataFrame의 각 열에 맞는 
    SQL 데이터 타입을 매핑하여 테이블에 데이터를 삽입합니다.

    Args:
        database (str): 데이터베이스 이름.
        table_name (str): 데이터를 삽입할 테이블 이름.
        data (pandas.DataFrame): 삽입할 데이터를 담고 있는 pandas DataFrame.

    Returns:
        None: 데이터가 성공적으로 삽입되면 메시지를 출력합니다. 
              오류가 발생하거나 DataFrame이 비어있을 경우 예외(테이블 생성) 또는 경고 메시지를 출력합니다.
    """

    # username = 'signlab'
    # password = 'signlab123'
    # host = '172.16.18.11'  # 또는 서버의 IP 주소
    # port = 3306 # MariaDB의 기본 포트
    
    username = 'bwms_dba'  # 사용자 이름
    password = '!^admin1234^!'  # 비밀번호
    host = 'signlab.iptime.org'  # 서버 주소
    port = 20002  # 포트

    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')

    # 데이터 타입 매핑을 위한 딕셔너리 생성
    dtype_mapping = {}
    for column in data.columns:
        # 각 열의 데이터 타입을 pandas에서 감지하여 매핑
        if pd.api.types.is_integer_dtype(data[column]):
            dtype_mapping[column] = Integer()
        elif pd.api.types.is_float_dtype(data[column]):
            dtype_mapping[column] = Float()
        elif pd.api.types.is_bool_dtype(data[column]):
            dtype_mapping[column] = Boolean()
        elif pd.api.types.is_datetime64_any_dtype(data[column]):
            dtype_mapping[column] = DateTime()
        elif pd.api.types.is_string_dtype(data[column]):
            # 문자열의 경우 최대 길이를 계산하여 VARCHAR로 설정
            max_length = data[column].astype(str).map(len).max()
            dtype_mapping[column] = VARCHAR(min(max_length + 10, 255))  # 최대 255자로 제한
        else:
            # 기본적으로 모든 것을 VARCHAR로 설정
            dtype_mapping[column] = VARCHAR(255)

    # 데이터 프레임을 MariaDB에 적재합니다.
    try:
        # DataFrame 'data'가 비어있는지 확인
        if not data.empty:
            data.to_sql(table_name, con=engine, if_exists='append', index=False, dtype=dtype_mapping)
            print(f'DataFrame has been successfully loaded into {table_name} table in {database} database.')
        else:
            # 빈 데이터 프레임인 경우
            print("The DataFrame is empty. No data was loaded into the table.")
            
    except Exception as e:
        # 예외 발생 
        print(f"Failed to load data:{e}")