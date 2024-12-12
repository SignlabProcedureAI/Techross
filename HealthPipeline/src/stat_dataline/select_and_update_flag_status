def get_and_update_flag_status():
    """
    Retrieve the flag status data from the database, filter it based on conditions,
    and update the FLAG column for the filtered rows.

    Returns:
        pd.DataFrame: The filtered data before updating the FLAG column.
    """
    username = 'signlab'
    password = ''
    host = '172.16.18.11'  # 또는 서버의 IP 주소
    port = 3306 # MariaDB의 기본 포트

    # SQLAlchemy 엔진 생성
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
    
    query = f"""
    SELECT * FROM `{table_name}` 
    WHERE  `DATA_TIME` BETWEEN `IS_COMPLETE` != 0 AND `IS_PREPROCESSING` = 0;
    """
    
    # Pandas를 사용하여 데이터 프레임으로 로드
    df = pd.read_sql(query, engine)
    if df.empty:
        print("No data matching the conditions.")
        return None

    # Update FLAG column to 1 for the filtered rows
    update_query = f"""
    UPDATE `tc_flag_status`
    SET `IS_PREPROCESSING` = 1
    WHERE `DATA_TIME` BETWEEN  `IS_COMPLETE` != 0 AND `IS_PREPROCESSING` = 0;
    """

    # Execute the update query within a transaction
    with engine.begin() as connection:
        connection.execute(update_query)

    print("FLAG updated to 1 for the filtered data.")

    return filtered_df

