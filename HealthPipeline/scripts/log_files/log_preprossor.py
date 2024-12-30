import pandas as pd

def preprocess_log_file(file_path):
    """
    로그 파일을 읽고 전처리하여 데이터프레임으로 반환하는 함수입니다.

    Parameters:
        file_path (str): 전처리할 로그 파일의 경로
    
    Returns:
        pd.DataFrame: 로그 데이터를 전처리한 데이터프레임
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 로그 메시지에서 공백과 줄바꿈 제거
            line = line.strip()
            if line:
                # 각 로그 항목을 파싱하여 key-value 쌍으로 분리
                parsed_line = dict(item.split('=') for item in line.split(' | '))
                data.append(parsed_line)
    
    # 데이터 리스트를 데이터프레임으로 변환
    df = pd.DataFrame(data)

    return df