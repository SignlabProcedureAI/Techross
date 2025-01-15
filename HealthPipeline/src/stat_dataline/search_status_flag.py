# basic
import pandas as pd

# database
from sqlalchemy import create_engine

# module
from stat_dataline.flag_status_manager import FlagStatusManager 
from stat_dataline.db_engine import DatabaseEngine 

class ReferenceDateManager:
    """ [책임] 건전성 분석을 위한 날짜 추출을 관리
    """
    
    def __init__(self, flag_manager:FlagStatusManager):
        self.__flag_manager = flag_manager #[캡슐화] FlagManager 의존성 주입

    
    def get_reference_dates(self):
        """ [행동] 건전성 분석을 위한 날짜 목록을 반환
        """
        filtered_df = self.__flag_manager.get_and_update_falg_status()
        if filtered_df is None or filtered_df.empty:
            print("[ReferenceDateManager] No valid reference dates found.")
            return []

        reference_dates = filtered_df['REFERENCE_DT'].tolist()
        print("\n[건전성 분석을 위한 날짜 추출]...")
        print(f"{reference_dates}")

        return reference_dates


class ScheduledDataFetcher:
    """ [책임] 주어진 일정에 따라 데이터를 필터링하고 반환
    """
    
    def __init__(self, db_engine:DatabaseEngine, table_name):
        self.__engine = db_engine.engine # [캡슐화] DB 엔진 접근
        self.__table_name = table_name # [캡슐화] 테이블 이름

    
    def __fetch_data(self, start_time, end_time):
        """  [내부 전용 메서드] 주어진 시간 범위에서 데이터 조회
        """

        query = f"""
        SELECT * FROM `{self.__table_name}` 
        WHERE `REG_DATE` BETWEEN '{start_time}' AND '{end_time}' AND `FLAG` = 0;
        """
        df = pd.read_sql(query, self.__engine)

        return df

    
    def __update_flag(self, start_time, end_time):
        """ [내부 전용 메서드]
        """

        update_query = f"""
        UPDATE `{self.__table_name}`
        SET `FLAG` = 1
        WHERE `REG_DATE` BETWEEN '{start_time}' AND '{end_time}' AND `FLAG` = 0;
        """
        with self.__engine.begin() as connection:
            connection.execute(update_query)

    
    def fetch_data_on_schedule(self, start_time, end_time):
        """ [행동] 일정에 따라 데이터 필터링 및 FLAG 업데이트트
        """
        
        df = self.__fetch_data(start_time, end_time)
        if df.empty:
            print("[ScheduledDataFetcher] No data found for the given schedule.")
            return None
        
        self.__update_flag(start_time, end_time)
        
        return df
    

class DataFilterManager:
    """ [책임] 날짜를 기반으로 데이터 필터링 및 스케줄링 관리 
    """

    def __init__(self, reference_date_manager:ReferenceDateManager, scheduled_fetcher:ScheduledDataFetcher):
        self.__reference_date_manager = reference_date_manager  # [캡슐화] ReferenceDateManager 의존성
        self.__scheduled_fetcher = scheduled_fetcher  # [캡슐화] ScheduledDataFetcher 의존성

    
    def filter_by_flag_status(self):
        """ [행동] 날짜 목록을 기준으로 데이터를 필터링
        """

        reference_dates  = self.__reference_date_manager.get_reference_dates()
        if not reference_dates:
            print("[DataFilterManager] No reference dates available.")
            return None
        
        filtered_dataframes = []

        for date in reference_dates:
            start_time = pd.Timestamp(date)
            end_time = start_time + pd.Timedelta(days=1)

            filtered_data = self.__scheduled_fetcher.fetch_data_on_schedule(start_time, end_time)
            if filtered_data is not None:
                filtered_dataframes.append(filtered_data)

        print("\n[건전성 분석 데이터 리턴...]")
        return pd.concat(filtered_dataframes, ignore_index=True) if filtered_dataframes else None
    

class DataPipelineManager:
    """
    [Facade Pattern] 데이터 파이프라인 초기화 및 실행 관리
    """

    def __init__(self):
        """ [행동] 모든 객체 초기화 및 의존성 주입
        """
        # DB 엔진 생성
        self.__db_engine = DatabaseEngine(
            username='signlab',
            password='signlab123',
            host='172.16.18.11',
            port=3306,
            database='signlab'
        )

        # Flag 상태 매니저 생성
        self.__flag_manager = FlagStatusManager(
            engine = self.__db_engine.engine,
            table_name = 'tc_flag_status'
        )

        # Reference Date 매니저 생성
        self.__reference_date_manager = ReferenceDateManager(self.__flag_manager)

        # Scheduled Data Fetcher 생성
        self.__schedule_fetcher = ScheduledDataFetcher(
            db_engine=self.__db_engine,
            database='signlab',
            table_name='tc_ecs_data_flag'
        )

        # 데이터 필터링 매니저 생성
        self.__data_filter_manager = DataFilterManager(
            self.__reference_date_manager,
            self.__schedule_fetcher
        )

    
    def run_pipeline(self):
        """ [행동] 데이터 필터링 실행 및 결과 반환
        """
        print("[DataPipelineManager] 데이터 파이프라인 실행 시작...")
        result = self.__data_filter_manager.filter_by_flag_status()
        print("[DataPipelineManager] 데이터 파이프라인 실행 완료.")
        return result
    