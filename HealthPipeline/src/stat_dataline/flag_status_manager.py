# select_and_update_flag_status  → flag_status_manager로 모듈명 변경
import pandas as pd

class FlagStatusManager:
    """ [캡슐화] Flag 상태를 안전하게 조회 및 업데이트
    """

    def __init__(self, engine, table_name):
        self.__engine = engine # [캡슐화] 엔진 보호
        self.__table_name = table_name # [캡슐화] 테이블 이름 보호

    
    @property
    def table_name(self):
        """ [추상화] 테이블 이름 읽기 전용
        """
        return self.__table_name
    
    def __execute_query(self, query):
        """ [캡슐화] 쿼리를 안전하게 실행 (내부 전용)
        """
        with self.__engine.begin() as connection:
            connection.execute(query)


    def __fetch_data(self, query):
        """ [캡슐화] 쿼리 결과를 안전하게 가져오기 (내부 전용)
        """

        return pd.read_sql(query, self.engine)
    

    def get_flag_status(self):
        """ [내부 전용 메서드] Flag 상태 데이터 조회
        """

        query = f"""
        SELECT * FROM `{self.__table_name}` 
        WHERE `IS_COMPLETE` != 0 AND `IS_PREPROCESSING` = 0;
        """
        
        return self.__fetch_data(query)
        

    def update_flag_status(self):
        """ [내부 전용 메서드] Flag 상태를 업데이트트
        """
        
        query = f"""
        UPDATE `{self.__table_name}`
        SET `IS_PREPROCESSING` = 1
        WHERE `IS_COMPLETE` != 0 AND `IS_PREPROCESSING` = 0;
        """

        self.__execute_query(query)
        print("FLAG updated to 1 for the filtered data.")

    
    def get_and_update_falg_status(self):
        """ [행동] 데이터 조회 및 업데이트 실행
        """
        df = self.get_flag_status()
        if df.empty:
            print("No data matching the conditions.")
            return None
        self.update_flag_status()
        return df
        
        

    
