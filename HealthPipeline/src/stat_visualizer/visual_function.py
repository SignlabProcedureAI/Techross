#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Goals: 통계적 허용 변화량 찾기

def seaborn_histplot_with_std_lines_and_return_limits(df,columns):
   # 올바른 방법으로 figure 사이즈 설정
    plt.figure(figsize=(20, 10))
    sns.histplot(x=columns,data=df, kde=True,bins=5,alpha=0.4,color='blue',edgecolor='black')
    
    # 평균과 표준편차 계산
    mean = df[columns].mean()
    std = df[columns].std()
    
    # 상한선과 하한선 (평균 ± 2*표준편차)
    upper_limit = mean + 1.96 * std
    lower_limit = mean - 1.96 * std

    # axline으로 상한선과 하한선 표시
    plt.axvline(upper_limit, color='r', linestyle='--', label='Upper Limit (Mean + 1.96*STD)')
    plt.axvline(lower_limit, color='b', linestyle='--', label='Lower Limit (Mean - 1.96*STD)')

    # 범례 추가
    plt.legend()
    
    # 상한선과 하한선 반환
    return upper_limit, lower_limit


def plot_system_health(data):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health TREND', color='blue',linestyle='--')
    #sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health Trend', color='orange',linestyle='--')
    
     # 모든 x 위치에 'X' 표시 추가
    plt.scatter(data['DATA_INDEX'], data['HEALTH_TREND'], marker='x', color='red', label='All Points')
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Comprehensive System Health Trend')
    plt.ylim(0,100)
    plt.xlabel('Operation Index')
    plt.ylabel('Health Trend')
    
    # 플롯 표시
    plt.show()
    
    
def plot_sensor(data,col):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='DATA_INDEX',y=col,data=data,label=f'{col}', color='blue',linestyle='--')
    #sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health Trend', color='orange',linestyle='--')
    
     # 모든 x 위치에 'X' 표시 추가
    plt.scatter(data['DATA_INDEX'], data[col], marker='x', color='red', label='All Points')
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Sensor Plot')
    plt.xlabel('DATA INDEX')
    plt.ylabel(f'{col}')
    
    # 플롯 표시
    plt.show()
    
def save_plot_sensor(data,col,ship_id,op_index,file_path,abnormal=False):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='DATA_INDEX',y=col,data=data,label=f'{col}', color='blue',linestyle='--')
    #sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health Trend', color='orange',linestyle='--')
    
    if abnormal:
        # 모든 x 위치에 'X' 표시 추가
        plt.scatter(data['DATA_INDEX'][data['State'] =='Abnormal'], data[col][data['State'] == 'Abnormal'], color='red', label='Abnormal', s=100, zorder=5)
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Sensor Plot')
    plt.xlabel('DATA INDEX')
    plt.ylabel(f'{col}')
    
    # 플롯 표시
    plt.savefig(f'{file_path}\\techross_{col}_{ship_id}_{op_index}.png', dpi=300)
    
    
    
def plot_sensor_abnormal(data,col):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='DATA_INDEX',y=col,data=data,label=f'{col}', color='blue',linestyle='--')
    #sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health Trend', color='orange',linestyle='--')
    
    plt.scatter(data['DATA_INDEX'][data['State'] =='Abnormal'], data[col][data['State'] == 'Abnormal'], color='red', label='Abnormal', s=100, zorder=5)
        
     # 모든 x 위치에 'X' 표시 추가
    #plt.scatter(data['DATA_INDEX'], data[col], marker='x', color='red', label='All Points')
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Sensor Plot')
    plt.xlabel('DATA INDEX')
    plt.ylabel(f'{col}')
    
    # 플롯 표시
    plt.show()
    
    
def plot_current_system_health(data):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(x='DATA_INDEX',y='ELECTRODE_EFFICIENCY_TREND',data=data,label='ELECTRODE_EFFICIENCY_TREND', color='blue',linestyle='--')
    #sns.lineplot(x='DATA_INDEX',y='HEALTH_TREND',data=data,label='Health Trend', color='orange',linestyle='--')
    
     # 모든 x 위치에 'X' 표시 추가
    plt.scatter(data['DATA_INDEX'], data['ELECTRODE_EFFICIENCY_TREND'], marker='x', color='red', label='All Points')
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Comprehensive System Health Trend')
    plt.ylim(0,-100)
    plt.xlabel('Operation Index')
    plt.ylabel('Health Trend')
    
    # 플롯 표시
    plt.show()

    
def line_plots_twinx(df,col):
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 그래프 그리기
    ax.set_xlabel('DATA_INDEX')
    ax.set_ylabel('HEALTH SCORE', color='blue')
    ax.plot(df['DATA_INDEX'], df['HEALTH_RATIO'], label='HEALTH_RATIO', color='blue', linestyle='--')
    ax.scatter(df['DATA_INDEX'], df['HEALTH_RATIO'], marker='x', color='red', label='All Points')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.tick_params(axis='x', labelcolor='blue', rotation=90)
    ax.set_ylim(0,100)
    
    ax2 = ax.twinx()  # 'twinx' 메서드를 사용하여 x축을 공유하는 새로운 y축 생성
    ax2.set_ylabel(f'{col} TREND', color='orange')
    ax2.plot(df['DATA_INDEX'],df[col], label=f'{col} TREND',color='orange') # 레이블 추가
    ax2.tick_params(axis='y', labelcolor='orange')
    #ax2.set_ylim(0,45)
    

    plt.title('HEALTH SCORE')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()  # 레이블이나 제목이 잘리지 않도록 레이아웃을 조정합니다.
    
    # 그래프 출력
    plt.show()
    
def plot_corr_scatter(data,col):
     # 플롯 스타일 설정
    sns.set_style("whitegrid")
    
     # 선형 플롯 생성
    plt.figure(figsize=(10, 6))
    
    # 모든 x 위치에 'X' 표시 추가
    plt.scatter(data[col], data['FTS'], marker='x', color='red', label='All Points')
    
    # 중복된 라벨을 제거하기 위해 범례 항목 필터링
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # 제목과 축 라벨 추가
    plt.title('Comprehensive System Health Trend')
    plt.xlabel('Operation Index')
    plt.ylabel('Health Trend')
    
    # 플롯 표시
    plt.show()
    
    
def line_plots_time_offset(df,col):
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 그래프 그리기
    ax.set_xlabel('Index')
    ax.set_ylabel('TRO', color='blue')
    ax.plot(df['DATA_INDEX'], df[col], label='TRO', color='blue')# 레이블 추가
    ax.tick_params(axis='y', labelcolor='blue')
    ax.tick_params(axis='x', labelcolor='blue', rotation=90)
    
    ax2 = ax.twinx()  # 'twinx' 메서드를 사용하여 x축을 공유하는 새로운 y축 생성
    ax2.set_ylabel('Predict_TRO', color='orange')
    ax2.plot(df['DATA_INDEX'],df['CURRENT'], label='CURRENT',color='orange') # 레이블 추가
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # 비정상 데이터(Steep Decline)은 빨간색 점으로 표시 / 비정상 데이터(Slowly Decreasing)은 보라색 점으로 표시 
    ax.scatter(df['DATA_INDEX'][df['TIME_OFFSET'] == 1], df[col][df['TIME_OFFSET'] == 1], color='red', label='TIME OFFSET', s=100, zorder=5)
  
    # OP_INDEX가 변경되는 지점 찾기(여기선 사용하지 않음)
    op_index_changes = df['OP_INDEX'].diff() != 0  # 이 부분은 각 행이 OP_INDEX가 변경되었는지를 나타내는 부울 시리즈를 생성합니다.
    change_points = df[op_index_changes]['DATA_INDEX']  # 변경된 OP_INDEX를 가진 행의 DATA_TIME을 가져옵니다.
    
    # OP_INDEX가 변경될 때마다 수직선 그리기(여기선 사용하지 않음)
    for change_point in change_points:
        ax.axvline(x=change_point, color='gray', linestyle='--', linewidth=2, zorder=2)  # zorder는 선이 다른 그래프 요소보다 뒤에 그려지도록 합니다.
        # OP_INDEX 값을 가져와서 DATA_TIME 추가
        op_index_value = df.loc[df['DATA_INDEX'] == change_point, 'OP_INDEX'].iloc[0]  # 변경 지점의 OP_INDEX 값
        ax.text(change_point, df[col].max(), f'OP:{op_index_value}', rotation=90, verticalalignment='center', fontsize=9, backgroundcolor='white', zorder=5)
    
    # 축 이름 및 제목 추가
   #plt.xlabel('Time')
    plt.xticks([])
    plt.ylabel(col)
    #plt.ylim(0,9)
    plt.title('Visualization of Normal vs Abnormal')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()  # 레이블이나 제목이 잘리지 않도록 레이아웃을 조정합니다.
    
    # 그래프 출력
    plt.show()
    
def line_plots(df,col):
    # 그래프 그리기
    plt.figure(figsize=(15, 12))
    plt.plot(df['DATA_TIME'], df[col], label='Normal', color='blue')
    #plt.plot(df['DATA_TIME'],df['TRO_PRED'], label='PRED',color='orange')
    
    # 비정상 데이터(Steep Decline)은 빨간색 점으로 표시 / 비정상 데이터(Slowly Decreasing)은 보라색 점으로 표시 
    plt.scatter(df['DATA_TIME'][df['STEEP_LABEL'] == 1], df[col][df['STEEP_LABEL'] == 1], color='purple', label='Steep Decline', s=100, zorder=5)
    plt.scatter(df['DATA_TIME'][df['OUT_OF_WATER_STEEP'] == 1], df[col][df['OUT_OF_WATER_STEEP'] == 1], color='red', label='OUT_OF_WATER_STEEP', s=100, zorder=5)
    plt.scatter(df['DATA_TIME'][df['SLOWLY_LABEL'] == 1], df[col][df['SLOWLY_LABEL'] == 1], color='yellow', label='Slowly Decreasing', s=100, zorder=5)
     #비정상 데이터(Steep Decline)은 빨간색 점으로 표시 / 비정상 데이터(Slowly Decreasing)은 보라색 점으로 표시 
    plt.scatter(df['DATA_TIME'][df['HUNTING'] == 1], df[col][df['HUNTING'] == 1], color='blue', label='HUNTING', s=100, zorder=5)
    plt.scatter(df['DATA_TIME'][df['TIME_OFFSET'] == 1], df[col][df['TIME_OFFSET'] == 1], color='orange', label='TIME_OFFSET', s=100, zorder=5)
    
    # OP_INDEX가 변경되는 지점 찾기(여기선 사용하지 않음)
    op_index_changes = df['OP_INDEX'].diff() != 0  # 이 부분은 각 행이 OP_INDEX가 변경되었는지를 나타내는 부울 시리즈를 생성합니다.
    change_points = df[op_index_changes]['DATA_TIME']  # 변경된 OP_INDEX를 가진 행의 DATA_TIME을 가져옵니다.
    
    # OP_INDEX가 변경될 때마다 수직선 그리기(여기선 사용하지 않음)
    for change_point in change_points:
        plt.axvline(x=change_point, color='gray', linestyle='--', linewidth=2, zorder=2)  # zorder는 선이 다른 그래프 요소보다 뒤에 그려지도록 합니다.
        # OP_INDEX 값을 가져와서 DATA_TIME 추가
        op_index_value = df.loc[df['DATA_TIME'] == change_point, 'OP_INDEX'].iloc[0]  # 변경 지점의 OP_INDEX 값
        plt.text(change_point, df[col].max(), f'OP:{op_index_value}', rotation=90, verticalalignment='center', fontsize=9, backgroundcolor='white', zorder=5)
    
    # 축 이름 및 제목 추가
   #plt.xlabel('Time')
    plt.xticks([])
    plt.ylabel(col)
    #plt.ylim(0,9)
    plt.title('Visualization of Normal vs Abnormal')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.tight_layout()  # 레이블이나 제목이 잘리지 않도록 레이아웃을 조정합니다.
    
    # 그래프 출력
    plt.show()