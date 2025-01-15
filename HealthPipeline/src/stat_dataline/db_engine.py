from sqlalchemy import create_engine

class DatabaseEngine:
    """ [캡슐화] 데이터베이스 엔진 생성 및 관리
    """

    def __init__(self, username, password, host, port, database):
        self.__engine = self.__create_engine(username, password, host, port, database)


    def __create_engine(self, username, password, host, port, database):
        """ [캡슐화] 안전하게 엔진 생성 (외부 접근 차단)
        """

        return create_engine(
             f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        )

        
    @property
    def engine(self):
        """ [추상화] 안전한 엔진 접근
        """
        return self.__engine
    
    