# basic
import pandas as pd
import numpy as np
import os

# visualize
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns



# module
from prep.preprocessing import apply_preprocessing_fuction


# 인터랙티브 모드 비활성화
plt.ioff()

# 비대화형 백엔드 설정
matplotlib.use('Agg')  


def find_folder(file_path):

    # 파일의 디렉토리 경로 추출
    directory = os.path.dirname(file_path)

    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

# 이상치 범위 dict 생성
outlier_ballast_dict = {'CSU':[0,49.46],'STS':[0,33.29],'FTS':[0,39.24],'FMU':[286,2933],'TRO':[0,8],'CURRENT':[0,18790],'VOLTAGE':[3.0,4.7]}
outlier_deballast_dict = {'CSU':[0,49.46],'STS':[0,33.29],'FMU':[286,2933],'TRO':[0,1.79], 'ANU':[0,1320]}

def select_data_variable(original_data, data):
    drop_columns = ['vessel', 'tons', 'tons_category']
    
    original_data = original_data.drop(columns=drop_columns)
    data = data.drop(columns=drop_columns)
    
    original_data = original_data.rename({'ship_name':'SHIP_NAME'},axis=1)
    data = data.rename({'ship_name':'SHIP_NAME'},axis=1)
    
    return original_data, data


def plot_histograms_with_noise(df, ship_id, op_index, section, str_op_type):
    """
    Plot histograms with hue for the specified columns of the DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - columns: list of column names (variables) to plot
    - hue_column: column to be used as hue
    """
    # 색상 설정
    fill_color = "#FF69B4"
    line_color = "#FFA500"   
    
    # 스타일 및 그리드 설정
    sns.set(style="whitegrid")
    
    # 그림 설정
    fig = plt.figure(figsize=(15, 8))
    
    # 오퍼레이션 구분
    op_type = df['OP_TYPE'].iloc[0]

    # 범례를 미리 저장하기 위한 변수
    handles, labels = None, None
    
    # Seaborn 팔레트 설정
    #palette = sns.color_palette('Set2')  # 색상 팔레트를 Set2로 설정
    palette = ['#2ca02c', '#1f77b4']
    # Ballasting 경우
    
    if op_type!=2:
        columns=['CSU','STS','FTS','FMU','TRO','CURRENT','VOLTAGE']
        for i, column in enumerate(columns, start=1):
             # 모든 값이 0이면 건너뛰기
            # if df[column].eq(0).all():
            #     print(f"Skipping column {column} because all values are 0.")
            #     continue

            plt.subplot(2, 4, i)  # 2x2 subplot layout

            #sns.kdeplot(data=df, x=column, fill=True, color=fill_color, hue='state')  # Using kdeplot with hue
             
             # kdeplot을 각각의 레이블과 함께 호출
            sns.kdeplot(data=df[df['state'] == 'original'], x=column, fill=True, label='original', color=palette[0])
            sns.kdeplot(data=df[df['state'] == 'preprocessing'], x=column, fill=True, label='preprocessed', color=palette[1])

            # 현재 subplot의 Axes 객체를 가져옴
            ax = plt.gca()
            
            # 첫 번째 subplot에서 범례를 저장
            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

            # 범례가 있는 경우에만 제거
            if ax.legend_ is not None:
                ax.legend_.remove()

             # 데이터의 최소값과 최대값 계산
            data_min = df[column].min()
            data_max = df[column].max()

            # 데이터가 범위를 벗어나는지 확인
            lower_bound, upper_bound = outlier_ballast_dict[column]

            if data_min < lower_bound or data_max > upper_bound:

                # outlier 범위 표시
                plt.axvline(outlier_ballast_dict[column][0], color=line_color, linestyle='--', alpha=0.8)
                plt.axvline(outlier_ballast_dict[column][1], color=line_color, linestyle='--', alpha=0.8)


        # 전체 figure에 하나의 범례를 추가합니다.
        fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    else:
        columns=['CSU','STS','FMU','TRO','ANU']
        for i, column in enumerate(columns, start=1):
            plt.subplot(2, 3, i)  # 2x2 subplot layout

            #sns.kdeplot(data=df, x=column, fill=True, color=fill_color)  # Using kdeplot with hue
            # kdeplot을 각각의 레이블과 함께 호출
            sns.kdeplot(data=df[df['state'] == 'original'], x=column, fill=True, label='original')
            sns.kdeplot(data=df[df['state'] == 'preprocessing'], x=column, fill=True, label='preprocessing')
            
            # 현재 subplot의 Axes 객체를 가져옴
            ax = plt.gca()

             # 첫 번째 subplot에서 범례를 저장
            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

            # 범례가 있는 경우에만 제거
            if ax.legend_ is not None:
                ax.legend_.remove()

             # 데이터의 최소값과 최대값 계산
            data_min = df[column].min()
            data_max = df[column].max()

            # 데이터가 범위를 벗어나는지 확인
            lower_bound, upper_bound = outlier_deballast_dict[column]

            if data_min < lower_bound or data_max > upper_bound:

                # outlier 범위 표시
                plt.axvline(outlier_deballast_dict[column][0], color=line_color, linestyle='--', alpha=0.8)
                plt.axvline(outlier_deballast_dict[column][1], color=line_color, linestyle='--', alpha=0.8)


    plt.tight_layout()
   
    #if boa == 'before':
    file_path = f'D:\\bwms\\preprocessing\\{ship_id}\\{op_index}\\{ship_id}_{op_index}_{section}_noise_ba.png' 

    # 파일의 디렉토리 경로 추출
    find_folder(file_path)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # 메모리 해제

    # 이 부분 주석
    #else:
        #plt.savefig(f"D:\\bwms\\pract\\noise\\after\\{reg}_{idx}_{str_op_type}.png", dpi=300, bbox_inches='tight')
    #plt.show()
  


def plot_bar_with_operation(original_data,ship_id, op_index, section, op_type):

    # 전체 데이터 합산
    total_value = original_data['DATA_INDEX'].count()

    # 원본 데이터 오퍼레이션 삭제
    top_60_value =  original_data['DATA_INDEX'][original_data['DATA_INDEX']<=60].count()

    # 삭제 후 오퍼레이션 
    total_index_after= total_value - top_60_value

    # 중첩 바 그래프 그리기
    plt.figure(figsize=(8, 6))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # x축 위치 설정
    x_positions = [0, 1]  # 왼쪽에 첫 번째 막대, 오른쪽에 두 번째 막대 위치

    # 전체 데이터에 대한 바
    plt.bar(x_positions[0], total_value, label='Total (All DATA_INDEX)', color='#7F9DB9', width=0.4, alpha=0.6)

    # 상위 60개 데이터에 대한 바
    plt.bar(x_positions[0], top_60_value, label='Top 60 DATA_INDEX', color='#A0A3A4', width=0.4, alpha=0.6)
    plt.bar(x_positions[0], top_60_value, color='None', edgecolor='#BD7677', hatch='//', width=0.4, alpha=0.6)
    plt.text(x=x_positions[0], y=top_60_value/2, s='Operation Data to be deleted', fontsize=10, color='white', ha='center')

    # 삭제 후 남은 데이터에 대한 바 (오른쪽 막대)
    plt.bar(x_positions[1], total_index_after, label='Remaining (After preprocessing)', color='#BD7677', width=0.4, alpha=0.7)

    # 기준선 추가
    plt.axhline(y=total_index_after, color='red', linestyle='-', linewidth=1, alpha=0.7)
    plt.text(x=x_positions[1], y=total_index_after+5, s=f"After preprocessing: {total_index_after}", color='#BD7677', fontsize=12, ha='center')

    # 주석 추가
    plt.text(x=x_positions[1], y=total_index_after/2, s='Deleted Data Remaining', color='white', fontsize=10, ha='center')

    # 배경색 설정
    plt.gca().set_facecolor('#F7F7F7')

    # x축 눈금 설정
    plt.xticks(x_positions, ['Original Data', 'After Preprocessing'])

    # 라벨과 제목 추가
    plt.ylabel('Values', fontsize=12)

    plt.tight_layout()

    file_path = f"D:\\bwms\\preprocessing\\{ship_id}\\{op_index}\\{ship_id}_{op_index}_{section}_operation_ba.png"

    # 파일의 디렉토리 경로 추출
    find_folder(file_path)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # 메모리 해제
    #plt.show()
    

def plot_pie_with_duplication(original_data, ship_id, op_index, section, op_type):

    # 결측치 비율 계산
    duplicates = original_data[original_data.duplicated(subset=['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME'], keep='first')]
    data = original_data[~original_data.duplicated(subset=['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME'], keep='first')]

    before_ratio = len(duplicates) 
    after_ratio = len(original_data) - before_ratio

    # 비율 비교 시각화
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    #plt.pie([before_ratio, len(original_data) - before_ratio], labels=['Duplication', 'Available'], autopct='%1.1f%%', colors=['darkblue', 'darkgreen'])
    plt.pie([before_ratio, len(original_data) - before_ratio], 
        labels=['Duplication', 'Available'], 
        autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * (before_ratio + len(original_data) - before_ratio) / 100), 
        colors=['#A020F0', '#40E0D0'])
    
    plt.title('Before Removal')

    plt.subplot(1, 2, 2)
    # plt.pie([after_ratio,  len(original_data) - after_ratio], labels=['Duplication', 'Available'], autopct='%1.1f%%', colors=['darkblue', 'darkgreen'])
    plt.pie([after_ratio, len(data) - after_ratio], 
        labels=['Available', 'Duplication'], 
        autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * (after_ratio + len(data) - after_ratio) / 100),
        colors=['#40E0D0', '#A020F0'])
    plt.title('After Removal')

    plt.tight_layout()

    file_path = f"D:\\bwms\\preprocessing\\{ship_id}\\{op_index}\\{ship_id}_{op_index}_{section}_duplication_ba.png"

    find_folder(file_path)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # 메모리 해제
    #plt.show()
    


def plot_double_bar_with_missing(original_data, ship_id, op_index, section, op_type): 

    # 각 변수의 총 카운트와 결측치 수 계산
    total_counts = original_data.count()
    missing_counts = original_data.isnull().sum()
    adjusted_counts = total_counts - missing_counts  # 결측치를 포함한 총 카운트

    # 그래프 설정
    fig, ax = plt.subplots(figsize=(12, 8))

    # 왼쪽 막대: 결측치 제외한 카운트와 결측치
    ax.barh(total_counts.index, total_counts, color='#3A9DBF', label='Total Count (No Missing)', align='center')
    ax.barh(total_counts.index, missing_counts, color='#D3D3D3', label='Missing Count', align='center')

    # 오른쪽 막대: 결측치를 포함한 총 카운트
    ax.barh(total_counts.index, adjusted_counts, left= -adjusted_counts, color='#6A1B9A', label='Total Count (After removing the Missing )', align='center', alpha=0.8)

    # 중앙 축 설정
    ax.axvline(x=0, color='black',linewidth=0.5)
    # 현재의 x축 눈금 값 가져오기
    current_ticks = ax.get_xticks()

    # 모든 눈금 값을 양수로 변환하여 라벨 설정
    ax.set_xticklabels([str(abs(int(tick))) for tick in current_ticks])

    # 레이블과 타이틀 설정
    ax.set_xlabel('Count')
    ax.set_ylabel('Variables')
    ax.set_title('Comparison of Counts with and without Missing Data')
    ax.legend()

    # 그래프 보여주기
    plt.tight_layout()

    file_path = f"D:\\bwms\\preprocessing\\{ship_id}\\{op_index}\\{ship_id}_{op_index}_{section}_missing_ba.png"

    find_folder(file_path)

    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # 메모리 해제
    #plt.show()
