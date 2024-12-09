#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import pickle

def load_model(filename):
    with open(filename, 'rb') as file:  # 'rb'는 바이너리 읽기 모드를 의미합니다.
        return pickle.load(file)
    
   
def calculate_health_ratio(data,col,arange_val):

    data['Allowable_error'] =data[col] * arange_val

    data['health ratio'] = np.round(data['diff'] / data['Allowable_error'],2) * 10
    
    return data

def health_ratio(data,acceptable_range):
    data['health ratio']=np.round((data['diff']/acceptable_range),2) *10


# In[ ]:

def value_to_percentage_cr(data):
    """
    주어진 값을 0 ~ max_value 범위의 백분율로 변환합니다.

    :param value: 실시간으로 들어오는 값 (0~2500 사이)
    :param max_value: 가능한 최대값 (기본값 2500)
    :return: 주어진 값을 백분율로 변환한 값
    """
    data['health percentage']= data['health ratio']  # 값의 백분율 계산
    
    return data


# In[ ]:

def make_rolling_mean(data,col,window_size):
    
    # 데이터 정리
    data.sort_index(inplace=True)
    
    # 이동 평균 생성
    data['rolling_mean']=data[col].rolling(window=window_size).mean()
    
    # 결측치 제거
    data.dropna(inplace=True)
    
    # 정제된 함수 반환 
    return data


def apply_current_model_predict(data,model,drop_column):
    system_data=data.copy()
    
    X=data[['CSU','STS','FTS','FMU','TRO','CURRENT','tons','VOLTAGE']]
    y=data[drop_column]
    
    independent_variable=X.drop(columns=drop_column)
    dependent_variable=data[drop_column]
        
    y_pred=model.predict(independent_variable)
    
    system_data['PRED']=y_pred
    
    system_data['diff']=system_data['PRED']-system_data[drop_column]
    
    return system_data

        
def group_dataset(data):
    
    # 고유 ship_id, op_index
    groupby_dataset=data.groupby(['SHIP_ID','OP_INDEX']).count().sort_values(by='OP_TYPE',ascending=False)['OP_TYPE'].to_frame()
    # 작동 시 200분 이상 작동한 데이터 셋 추출 
    groupby_dataset=groupby_dataset[groupby_dataset['OP_TYPE']>100]
    # 데이터 인덱스 설정
    groupby_index=groupby_dataset.index

    return groupby_index

def make_health_dataset(data,model,drop_column):
    
    # 데이터 셋 리스트
    datasets = []
    
    # 고유 인덱스 함수 사용
    groupby_index=group_dataset(data)
    
    for i in groupby_index:
        # 배 아이디
        ship_id=i[0]
        op_index=i[1]
        
        system_dataset=data[(data['SHIP_ID']==ship_id) & (data['OP_INDEX']==op_index)]
        system_model_dataset=system_dataset[['FMU', 'TRO','CURRENT','CSU','STS','tons','FTS']]
    
        independent_variable=system_model_dataset.drop(columns=drop_column)
        dependent_variable=system_model_dataset[drop_column]
        
        y_pred=model.predict(independent_variable)
    
        verification_dataset=make_accuracy_dataframe(independent_variable,dependent_variable,y_pred,drop_column)
        
        datasets.append(verification_dataset)
        
    # 모든 데이터셋을 하나로 합치기
    combined_data = pd.concat(datasets, ignore_index=True)
    
    return combined_data

def load_model(filename):
    with open(filename, 'rb') as file:  # 'rb'는 바이너리 읽기 모드를 의미합니다.
        return pickle.load(file)
    
    # Goals: 오류 데이터 테스트
def generate_bad_data(data,col):
    # 가중치 초기값 설정
    i=0
    
    for index,row in data.iterrows():
        if index>=350:
            # 현재 전류 값 
            current=row[col]
            # 현재 전류 값에서 가중치 곱한 값을 대입
            data.at[index, col] =current + (1.5 * i)
            # 가중치 증가
            i+=1
        
    return data

def make_accuracy_dataframe(X_test,y_test,y_pred,col):
    
    # iNDEX Number Map을 위해 인덱스 재 설정
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # 예측 값과 실제 값의 차이 계산
    X_test[col]=y_test
    X_test['pred']=y_pred
    
    # Diff 변수 생성
    X_test['diff']=X_test['pred']-X_test[col]
    
    diff_dataset=X_test
    
    return diff_dataset
