# basic
import numpy as np
import pandas as pd

# time
from datetime import datetime

# module.dataline
from prep_dataline.select_dataset import get_dataframe_from_database
from prep_dataline.load_database import load_database

def preprocess_data(data):
    
    # # 선박 필터링
    # selected_data = data[(data['SHIP_ID']==ship_id) & (data['OP_INDEX']==op_index) & (data['SECTION']==section)]
    
    # # 선박 데이터 개수
    # data_len = len(selected_data)
    
    # # 선박이 비어 있지 않을 때
    # if isinstance(selected_data, pd.DataFrame) & (data_len>=5):
    #     print(f'선박 데이터가 존재합니다. ( 데이터 길이: {data_len})')
    # else: # 선박이 조건을 만족하지 않을 때
    #     return None
    #     print("선박 데이터가 존재하지 않습니다.")
    
    # optpye 기준 데이터 필터링
    selected_data = select_col_based_optype(data)
    
    # 데이터 정리
    organized_data = organize_data(selected_data)
    
    return organized_data



def select_col_based_optype(data):
    # optype 추출
    op_type = data['OP_TYPE'].iloc[0]
    
    # Deballast 경우
    if op_type==2:
        # data = data[['ship_name','vessel','tons','tons_category','SHIP_ID','OP_INDEX','SECTION','OP_TYPE','VOYAGE','DATA_TIME','DATA_INDEX','CSU','STS','FMU','TRO','ANU']]
        data = data[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FMU','TRO','ANU']]
    else: # ballast 경우
        #data = data[['ship_name','vessel','tons','tons_category','SHIP_ID','OP_INDEX','SECTION','OP_TYPE','VOYAGE','DATA_TIME','DATA_INDEX','CSU','STS','FMU','FTS','TRO','CURRENT','VOLTAGE','RATE']]
        data = data[['SHIP_ID','OP_INDEX','SECTION','OP_TYPE','DATA_TIME','DATA_INDEX','CSU','STS','FMU','FTS','TRO','CURRENT','VOLTAGE','RATE','START_TIME', 'END_TIME', 'RUNNING_TIME']]
    return data



def organize_data(data):
    # 순위 정렬
    data = data.sort_values(by='DATA_INDEX')
    
    # 인덱스 재정렬
    data = data.reset_index(drop=True)
    
    return data



def drop_duplicate_values(data):
    
    indicator_data = data.copy()
    indicator_data['duplication'] = False
    
    if data is None:
        return None
    else:
        # 중복 행 식별 - 'SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME' 컬럼 기준
        columns_to_check = ['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME']
        duplicates = data.duplicated(subset=columns_to_check, keep='first')
        
        # 중복 데이터 인덱스 추출
        duplicates_index = data[duplicates].index

        # 모든 중복 행 제거
        sensor_no_duplicates = data[~duplicates]

        #Duplicated row Length
        duplicated_len = len(sensor_no_duplicates)
        
        # 중복 행 개수
        dulication_len = len(data) - duplicated_len

        # Source Data Length
        source_len = len(data)

        #Duplicated row ratio
        ratio=np.round((duplicated_len/source_len)*100,2)

        # print(f"중복 제거 후 데이터 개수: {len(sensor_no_duplicates)}")
        # print('Ratio after duplicated removal: {}%'.format(ratio))
        
        indicator_data.loc[duplicates_index,'duplication'] = True
        
        duplication_text = f'전체 데이터의 {np.round(100-ratio,2)}%가 중복,  중복 데이터를 제거함'
        
    return sensor_no_duplicates, indicator_data, duplication_text, dulication_len



def remove_missing_values(df, indicator_data):
    
    indicator_data['missing_value']=False
    
    if (df is None) or len(df)==0:
        return None
    else:
        df = df.sort_values(['SHIP_ID','OP_INDEX','SECTION','DATA_TIME'])

        df.replace('\\N', pd.NA, inplace=True)
        
        # 결측치 인덱스 추출
        subset_columns = indicator_data.columns.difference(['VOYAGE'])
        
        missing_indices = indicator_data[indicator_data[subset_columns].isna().any(axis=1)].index
        indicator_data.loc[missing_indices,'duplication'] = True
        
        # Null values Check and drop
        remove_null_df = df.dropna(subset=df.columns.difference(['VOYAGE']),how='any')

        # source_len
        source_len = len(df)

        #remove null val data len
        remove_null = len(remove_null_df)

        # null data len
        null_len = source_len - remove_null

        #ratio
        ratio = np.round((remove_null/source_len)*100,2)

        # print(f"결측치 제거 후 데이터 개수: {remove_null}")
        # print('null value removed data: {}%'.format(ratio))
        
        missing_values_text = f'전체 데이터의 {np.round(100-ratio,2)}%가 결측치, 결측치 데이터를 제거함'
        
    return remove_null_df, indicator_data, missing_values_text, null_len




def remove_negative_values(data):
    
    remove_negative_data = data[(data[col]>=0)]

    source_data=len(data)
    remove_data=len(remove_negative_data)
    
    ratio=np.round((remove_data/source_data)*100,2)

    # print('outlier value removed data({}의 음수 값 제거 후 데이터): {}'.format(col,remove_data))
    # print('Ratio after remove outlier: {}%'.format(ratio))
    # print('\n')




def remove_previous_operating_time(data, indicator_data):
    
    if data is None or len(data)==0:
        return None
    else:
        # float → int 변환
        data['DATA_INDEX'] = data['DATA_INDEX'].astype(int)
        drop_index_original = data[data['DATA_INDEX']<=60].index

        # 원본 데이터 개수 저장
        source_len = len(data)
            
        # ~60분 데이터 제거
        data = data[data['DATA_INDEX']>60]

        # print(f"오퍼레이션 제거 후 데이터 개수: {len(data)}")
        
        indicator_data['operation_time'] = False
        
        drop_index = indicator_data[indicator_data['DATA_INDEX']<=60].index

        indicator_data.loc[drop_index,'operation_time'] = True
        
        ratio = np.round((len(drop_index_original)/source_len)*100,2) # 60분 이하 데이터 삭제 후 data와 비율을 구했음 → 수정 완료
        
        operation_values_text = f'전체 데이터의 {ratio}%에 해당하는 시작 후 60분 데이터를 제거'
        
        #print(f'drop_index_original:{len(drop_index_original)}')
        
    return data, indicator_data, operation_values_text, len(drop_index_original)



def remove_quantile_outliers(data,indicator_data,col): 
    """
    이 함수는 데이터의 주어진 열(col)에 대해 5% 하위 및 95% 상위 분위수를 기준으로 이상치를 제거합니다.
    이를 통해 데이터의 극단적인 값을 제외시키고 더 안정적인 데이터셋을 확보할 수 있습니다.
    
    매개변수:
    - data: 이상치를 검사할 DataFrame.
    - col: 이상치를 검사할 열의 이름.
    
    반환값:
    - 이상치가 제거된 DataFrame.
    """
    if data is None or len(data)==0 :
        return None
    else:
        zero_count = (data[col]==0).sum()
        total_count = len(data[col])
        
        # 0의 비율이 90% 이상인지 확인
        zero_ratio = zero_count/total_count
        
        if zero_ratio >= 0.9:
            remove_data = 0
            # print(f"열 '{col}'은 90% 이상이 0입니다. 이 열을 패스합니다.")
            indicator_data[f'{col}_outlier']=False
            operation_values_text = f'{col} 센서 : 90% 이상 0.'
            return data, indicator_data, operation_values_text, remove_data 
        
        # 오퍼레이션 추출 deballst, ballast 구분 
        
        op_type = data['OP_TYPE'].iloc[0]
        
        # 이상치 범위 dict 생성
        outlier_ballast_dict = {'CSU':[0,49.46],'STS':[0,33.29],'FTS':[0,39.24],'FMU':[286,2933],'TRO':[0,8],'CURRENT':[0,18790],'VOLTAGE':[3.0,4.7]}
        outlier_deballast_dict = {'CSU':[0,49.46],'STS':[0,33.29],'FMU':[286,2933],'TRO':[0,1.79], 'ANU':[0,1320]}
        
        # ballast의 경우
        if op_type!=2:
            col_range = outlier_ballast_dict[col]
            
            top_95 = col_range[1]
            bottom_95=col_range[0]

            # print(' ')
            # print('TOP {}:{}: '.format(col,top_95))
            # print('BOTTOM {}:{}: '.format(col,bottom_95))

            remove_outlier = data[(data[col]>bottom_95) & (data[col]<top_95)]
            
            indicator_data[f'{col}_outlier'] = False
            drop_index = indicator_data[(indicator_data[col]<=bottom_95) | (indicator_data[col]>=top_95)].index
            indicator_data.loc[drop_index, f'{col}_outlier'] = True
            
            source_data=len(data)
            remove_data=len(remove_outlier)
            ratio=np.round((remove_data/source_data)*100,2)

            # print('outlier value removed data({}의 아웃라이어 제거 후 데이터): {}'.format(col,remove_data))
            # print('Ratio after remove outlier: {}%'.format(ratio))
            # print('\n')
            
            if len(drop_index)>0:
                operation_values_text = f'{col} 센서 제거량: {np.round(100-ratio,2)}%,  범위: {bottom_95} ~ {top_95} 미만 또는 초과,  이상치 데이터 제거'
                # operation_values_text = f'전체 데이터의 {np.round(100-ratio,2)}%가 {col} 센서의 {bottom_95} ~ {top_95} 범위를 벗어나 이상치 항목으로 확인되어, 이들 이상치 데이터를 제거함'
            else:
                operation_values_text = None
        
        # Deballast의 경우
        else:
            col_range = outlier_deballast_dict[col]
            
            top_95 = col_range[1]
            bottom_95=col_range[0]
            
            # print(' ')
            # print('TOP {}:{}: '.format(col,top_95))
            # print('BOTTOM {}:{}: '.format(col,bottom_95))

            remove_outlier = data[(data[col]>bottom_95) & (data[col]<top_95)]
            
            indicator_data[f'{col}_outlier'] = False
            drop_index = indicator_data[(indicator_data[col]<=bottom_95) | (indicator_data[col]>=top_95)].index
            indicator_data.loc[drop_index, f'{col}_outlier'] = True
            
            source_data=len(data)
            remove_data=len(remove_outlier)
            ratio=np.round((remove_data/source_data)*100,2)

            # print('outlier value removed data({}의 아웃라이어 제거 후 데이터): {}'.format(col,remove_data))
            # print('Ratio after remove outlier: {}%'.format(ratio))
            # print('\n')
            
            if len(drop_index)>0:
                operation_values_text = f'{col} 센서 제거량: {np.round(100-ratio,2)}%,  범위: {bottom_95} ~ {top_95} 미만 또는 초과,  이상치 데이터 제거'
                #operation_values_text = f'전체 데이터의 {np.round(100-ratio,2)}%가 {col} 센서의 {bottom_95} ~ {top_95} 범위를 벗어나 이상치 항목으로 확인되어, 이들 이상치 데이터를 제거함'
            else:
                operation_values_text = None
                
            # operation_values_text = f'전체 데이터의 {1-ratio}%가 {col} 센서의 {bottom_95} ~ {top_95} 범위를 벗어나 이상치 항목으로 확인되어, 이들 이상치 데이터를 제거함'
            
        if remove_outlier.empty:
            print("이상 값 제거 후 데이터가 비어 있습니다.")
            return None,indicator_data  
        else:
            #print("데이터가 존재합니다.")
            pass
        
    return remove_outlier,indicator_data, operation_values_text, (source_data-remove_data)



def apply_remove_outliers(data,indicator_data):
    
    text_dict = {}
    noise_sum = 0

    # optpye 추출
    op_type = data['OP_TYPE'].iloc[0]
    
    # ballast의 경우
    if op_type!=2:
        cols = ['CSU','STS','FTS','FMU','TRO','CURRENT','VOLTAGE']
    else: # Deballast의 경우
        cols = ['CSU','STS','FMU','TRO','ANU']
    
    for col in cols:
        data, indicator_data, operation_values_text, removed_len  = remove_quantile_outliers(data,indicator_data,col)
        text_dict[col] = operation_values_text
        noise_sum += removed_len
        # 아웃라이어 제거 후 데이터가 None이면 함수 종료
        if data is None:
            return None,indicator_data  
        
    return data, indicator_data, text_dict, noise_sum




def apply_preprocessing_fuction(ship_id, op_index, section, data):
    
    # 선박 선택
    sensor = preprocess_data(data)

    # 선박 개수 
    source_len = len(sensor)
    
    # 중복 값 제거
    sensor_no_duplicates,indicator_data, duplication_text, duplication_len = drop_duplicate_values(sensor)

    # 결측치 제거
    remove_null_df,indicator_data, missing_values_text, missing_len = remove_missing_values(sensor_no_duplicates,indicator_data)

    # 오퍼레이션 삭제
    removed_operation_data,indicator_data, operation_values_text, operation_len = remove_previous_operating_time(remove_null_df,indicator_data)

    # 이상 값 제거
    removed_outlier_data, indicator_data, text_list, noise_len = apply_remove_outliers(removed_operation_data,indicator_data)
    
    # 전처리 지표 데이터 정리
    op_type = indicator_data['OP_TYPE'].iloc[0]
    
    if op_type!=2:
        indicator_col_data = indicator_data[['SHIP_ID', 'OP_INDEX',
       'SECTION', 'OP_TYPE', 'DATA_TIME','DATA_INDEX', 'duplication',
       'missing_value', 'operation_time', 'CSU_outlier', 'STS_outlier','FTS_outlier',
       'FMU_outlier', 'TRO_outlier', 'CURRENT_outlier', 'VOLTAGE_outlier']]
        
    else:
        indicator_col_data = indicator_data[['SHIP_ID', 'OP_INDEX',
       'SECTION', 'OP_TYPE', 'DATA_TIME','DATA_INDEX', 'duplication',
       'missing_value', 'operation_time', 'CSU_outlier', 'STS_outlier',
       'FMU_outlier', 'TRO_outlier', 'ANU_outlier']]
        
    if removed_outlier_data is None:
        print(" 이상치 제거 후 데이터가 없어, 전처리 결과로 None을 반환합니다.")
        return sensor, None, None, None 
    
    organized_data = organize_data(removed_outlier_data)
    
    # 처리 후 개수 
    pre_len = len(organized_data)
    
    # 목록 데이터 생성
    indicator_df = process_preprocessed_list(indicator_col_data,source_len, pre_len)

    # 처리 개수 데이터 적재
    data_preprocessed = generate_preprocessing_count_dataframe(duplication_len, missing_len, operation_len, noise_len)
    
    # 설명 텍스트 생성
    text_dict = generate_supplementary_explanation_text(duplication_text, missing_values_text, operation_values_text, text_list, data_preprocessed)
    
    return sensor, organized_data, indicator_df, data_preprocessed, text_dict


def generate_supplementary_explanation_text(duplication_text, missing_values_text, operation_values_text, text_list, indicator_df):
    
    text_dict ={}
    
    noise = indicator_df['NOISE'].iloc[0] > 0
    missing = indicator_df['MISSING'].iloc[0] > 0
    duplication = indicator_df['DUPLICATE'].iloc[0] > 0
    operation = indicator_df['OPERATION'].iloc[0] >0 
    
    if noise:
        text_dict['noise'] = text_list
    else:
        text_dict['noise'] = None
        
    if missing:
        text_dict['missing'] = missing_values_text
    else:
        text_dict['missing'] = None
        
    if duplication:
        text_dict['duplication'] = duplication_text
    else:
        text_dict['duplication'] = None
        
    if operation:
        text_dict['operation'] = operation_values_text
    else:
        text_dict['missing'] = None
    
    return text_dict




def process_preprocessed_list(indicator_data, source_len, pre_len): 
    
    # 시간 타입으로 변경
    indicator_data = indicator_data.copy()
    
    indicator_data['DATA_TIME'] = pd.to_datetime(indicator_data['DATA_TIME'])
    #indicator_data['DATA_TIME'] = indicator_data['DATA_TIME'].dt.strftime('%Y-%m-%d')
    
    
    # 딕셔너리 생성하다
    indicator_dict = {'SHIP_ID':None, 'OP_INDEX':None, 'SECTION':None,  'OP_TYPE':None,  'DATA_COUNT':None, 'PRE_COUNT':None, 'START_DATE':None, 'END_DATE':None, 'REG_DATE':None}
                      
    # 고정 값 인서트
    indicator_dict['SHIP_ID'] = indicator_data.loc[0,'SHIP_ID']
    indicator_dict['OP_INDEX'] = indicator_data.loc[0,'OP_INDEX']
    indicator_dict['SECTION'] = indicator_data.loc[0,'SECTION']
    indicator_dict['OP_TYPE'] = indicator_data.loc[0,'OP_TYPE']
    
    # 처리 값 인서트
    # indicator_dict = generate_noise_col(indicator_data,indicator_dict)
    # indicator_dict = generate_missing_col(indicator_data,indicator_dict)
    # indicator_dict = generate_duplication_col(indicator_data,indicator_dict)
    # indicator_dict = generate_operation_time_col(indicator_data,indicator_dict)
    
    # 날짜 처리
    indicator_dict['START_DATE'] = indicator_data.loc[0,'DATA_TIME']
    indicator_dict['END_DATE'] = indicator_data.iloc[-1,4]
    indicator_dict['REG_DATE'] = datetime.now()

    # 개수 처리
    indicator_dict['DATA_COUNT'] = source_len
    indicator_dict['PRE_COUNT'] = pre_len

    return indicator_dict




def generate_noise_col(indicator_data,indicator_dict):
    op_type = indicator_data['OP_TYPE'].iloc[0]
    outlier_sum=0
    # 발라스트 경우
    if op_type!=2:
        cols = ['CSU','STS','FTS','FMU','TRO','CURRENT','VOLTAGE']
    # deballast의 경우    
    else:
        cols = ['CSU','STS','FMU','TRO','ANU']
    
    # 반복문을 이용한 합산
    for col in cols:
        outlier_sum += indicator_data[f"{col}_outlier"].sum()
    
    # 만약 합산 값이 0 이상이라면
    if outlier_sum > 0:
        indicator_dict['NOISE'] = True
    else:
        indicator_dict['NOISE'] = False
        
    return indicator_dict



def generate_missing_col(indicator_data,indicator_dict):
    
    if (indicator_data['missing_value'].sum())>0:
        indicator_dict['MISSING'] = True
    else:
        indicator_dict['MISSING'] = False
        
    return indicator_dict



def generate_duplication_col(indicator_data,indicator_dict):
    
    if (indicator_data['duplication'].sum())>0:
        indicator_dict['DUPLICATE'] = True
    else:
        indicator_dict['DUPLICATE'] = False
    
    return indicator_dict



def generate_operation_time_col(indicator_data,indicator_dict):
    
    if (indicator_data['operation_time'].sum())>0:
        indicator_dict['OPERATION'] = True
    else:
        indicator_dict['OPERATION'] = False    
        
    return indicator_dict


def generate_preprocessing_count_dataframe(duplication_len, missing_len, operation_len, noise_len):
    
    count_dict = [{'NOISE':noise_len ,'MISSING':missing_len, 'DUPLICATE':duplication_len, 'OPERATION':operation_len}]
    
    count_df = pd.DataFrame(count_dict)
    
    return count_df
    # print(count_dict)
    # load_database()