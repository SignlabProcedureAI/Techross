#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[ ]:


def load_model(filename):
    with open(filename, 'rb') as file:  # 'rb'는 바이너리 읽기 모드를 의미합니다.
        return pickle.load(file)


# In[ ]:


# Goals: 정확도 검토

def make_accuracy_dataframe(X_test,y_test,y_pred,col):
    
    # iNDEX Number Map을 위해 인덱스 재 설정
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # 예측 값과 실제 값의 차이 계산
    X_test[col]=y_test
    X_test['pred']=y_pred
    
    # Diff 변수 생성
    X_test['diff']=abs(X_test[col]-X_test['pred'])
    
    # 순서 재정렬
    #diff_dataset=X_test.sort_values(by='diff')
    
    diff_dataset=X_test
    return diff_dataset


# In[ ]:


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
        system_model_dataset=system_dataset[['CSU','STS','FTS','FMU','TRO','CURRENT']]
    
        independent_variable=system_model_dataset.drop(columns=drop_column)
        dependent_variable=system_model_dataset[drop_column]
        
        y_pred=model.predict(independent_variable)
    
        verification_dataset=make_accuracy_dataframe(independent_variable,dependent_variable,y_pred,drop_column)
        
        datasets.append(verification_dataset)
        
    # 모든 데이터셋을 하나로 합치기
    combined_data = pd.concat(datasets, ignore_index=True)
    
    return combined_data


def apply_model_predict(data,model,drop_column):
    system_data=data.copy()
    
    X=data[['CSU','STS','FTS','FMU','TRO','CURRENT']]
    y=data[drop_column]
    
    independent_variable=X.drop(columns=drop_column)
    dependent_variable=data[drop_column]
        
    y_pred=model.predict(independent_variable)
    
    system_data['PRED']=y_pred
    
    system_data['diff']=abs(system_data['PRED']-system_data[drop_column])
    
    return system_data
        
def apply_model_predict_fts(data,model,drop_column):
    
    system_data=data.copy()
    
    X = system_data[['CSU','STS','FTS','FMU','TRO','CURRENT','tons','VOLTAGE']]
    y = system_data[drop_column]
    
    independent_variable=X.drop(columns=drop_column)
        
    y_pred=model.predict(independent_variable)
    
    system_data['PRED']=y_pred
    
    system_data['diff']=abs(system_data['PRED']-system_data[drop_column])
    
    return system_data

def apply_model_predict_sts_remove(data,model,drop_column):
    system_data=data.copy()
    
    X=data[['STS','FTS','FMU','TRO','CURRENT']]
    y=data[drop_column]
    
    independent_variable=X.drop(columns=drop_column)
    dependent_variable=data[drop_column]
        
    y_pred=model.predict(independent_variable)
    
    system_data['PRED']=y_pred
    
    system_data['diff']=abs(system_data['PRED']-system_data[drop_column])
    
    return system_data

    
# In[ ]:


def health_ratio(data,acceptable_range):
    data['health ratio']=np.round((data['diff']/acceptable_range),2)


# In[ ]:


def generate_rolling_mean(data,col,window_size):
    
    # 데이터 정리
    data.sort_index(inplace=True)
    
    # 이동 평균 생성
    data['rolling_mean']=data[col].rolling(window=window_size).mean()
    
    # 결측치 제거
    data.dropna(inplace=True)
    
    # 정제된 함수 반환 
    return data


# In[ ]:


def value_to_percentage(data,col,max_value):
    """
    주어진 값을 0 ~ max_value 범위의 백분율로 변환합니다.

    :param value: 실시간으로 들어오는 값 (0~2500 사이)
    :param max_value: 가능한 최대값 (기본값 2500)
    :return: 주어진 값을 백분율로 변환한 값
    """
    #data['health ratio']=(data[col] / max_value) * 100  # 값의 백분율 계산
    
    #data['health percentage']=(data[col] / max_value) * 100  # 값의 백분율 계산
    
    data['health percentage']=(data['health ratio']*10.0) 
    
    data['health percentage']=data['health percentage'].apply(lambda x: 0 if x < 0 else x )
    
    return data


# In[ ]:


# 실시간 시각화를 위한 함수
def visualize_health(dataset):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    sns.lineplot(data=dataset, x=dataset.index, y='HEALTH_PRECENTAGE', color='blue', linewidth=2.5)
    sns.lineplot(data=dataset, x=dataset.index, y='HEALTH_TREND', color='green', linewidth=3.0, alpha=0.7)
    
    
    plt.ylim(0,100)
    plt.title('Line Plot for System Health', fontsize=16, fontweight='bold')
    plt.xlabel('Index', fontsize=14, fontweight='bold')
    plt.ylabel('System Health Indicators', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def visualize_total_health(dataset):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    sns.lineplot(data=dataset, x=dataset.index, y='CSU_HEALTH_PERCENTAGE', color='green', linewidth=3.0, alpha=0.7,label='CSU_HEALTH_PERCENTAGE Score')
    sns.lineplot(data=dataset, x=dataset.index, y='STS_HEALTH_PERCENTAGE', color='red', linewidth=3.0, alpha=0.7,label='STS_HEALTH_PERCENTAGE')
    #sns.lineplot(data=dataset, x=dataset.index, y='FTS_HEALTH_PERCENTAGE', color='blue', linewidth=3.0, alpha=0.7, label='FTS_HEALTH_PERCENTAGE')
    sns.lineplot(data=dataset, x=dataset.index, y='HEALTH_TREND', color='orange', linewidth=3.0, alpha=0.7,label='HEALTH_TREND')
   
    
    # Adding a horizontal line at y=100 and text
    plt.axhline(y=100, color='r', linestyle='--')
    #plt.text(x=20, y=100, s='Personal Health Index', color='black', verticalalignment='bottom')

    plt.ylim(0,100)
    plt.title('Line Plot for System Health', fontsize=16, fontweight='bold')
    plt.xlabel('Index', fontsize=14, fontweight='bold')
    plt.ylabel('System Health Indicators', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 실시간 시각화를 위한 함수
def visualize_health_scatter(dataset):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    #sns.lineplot(data=dataset, x=dataset.index, y='total health percentage', color='green', linewidth=3.0, alpha=0.7,label='Total Health Score')
    sns.lineplot(data=dataset, x=dataset.index, y='rolling_mean', color='green', linewidth=3.0, alpha=0.7,label='Total Health Score')
    #sns.lineplot(data=dataset, x=dataset.index, y='fts health percentage', color='blue', linewidth=3.0, alpha=0.7, label='FTS Health Score')
    #sns.lineplot(data=dataset, x=dataset.index, y='sts health percentage', color='orange', linewidth=3.0, alpha=0.7,label='STS Health Score')
    sns.lineplot(data=dataset, x=dataset.index, y='csu health percentage', color='purple', linewidth=3.0, alpha=0.7,label='CSU Health Score')
    
    # Adding a horizontal line at y=100 and text
    plt.axhline(y=100, color='r', linestyle='--')
    #plt.text(x=20, y=100, s='Personal Health Index', color='black', verticalalignment='bottom')

    plt.ylim(0,100)
    plt.title('Line Plot for System Health', fontsize=16, fontweight='bold')
    plt.xlabel('Index', fontsize=14, fontweight='bold')
    plt.ylabel('System Health Indicators', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_histograms(df, columns):
    """
    Plot histograms for the specified columns of the DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names (variables) to plot
    """
    sns.set_style("whitegrid")
    
    if len(columns) != 1:
        raise ValueError("Exactly 5 columns should be provided.")
    
    # 그림 설정
    plt.figure(figsize=(15, 10))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown','gray','gold']
    
    for i, column in enumerate(columns, 1):
        plt.subplot(2, 3, i)  # 2x3 subplot layout
        sns.histplot(df[column], kde=True, bins=30, color=colors[i])  # Using seaborn for a nicer visualization
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()

