# basic
import pandas as pd
import numpy as np
import pickle


# module.dataline
from time_dataline.select_dataset import get_dataframe_from_database

# visualize
import matplotlib.pyplot as plt  

# learning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# math
from scipy.stats import linregress
from statsmodels.tsa.seasonal import seasonal_decompose


# load ship history
ship_info = get_dataframe_from_database('ecs_dat1','shipinfo',all=True)

# 데이터 로드
og_df = get_dataframe_from_database('ecs','tc_ai_current_system_health',all=True) 

# In[3]:


def create_electrode_data():
    """ 데이터 로드 및 전처리
    """
    
    # 정제
    data  = og_df[og_df['ELECTRODE_EFFICIENCY'].notna()]
    data = data[(data['ELECTRODE_EFFICIENCY']>-100)]
    data = data.drop_duplicates()
    
    # 결측치 제거
#     data = data.dropna()

    return data



def preprocess_data(data,ship_id):

    # 선박 선택 
    electrod_df = data[data['SHIP_ID']==ship_id]

    # 순서 정렬
    electrod_df = electrod_df.sort_values(by=['DATA_TIME'])
    
    electrod_df['DATA_TIME'] = electrod_df['DATA_TIME'].dt.strftime('%Y-%m-%d %H:%M')
    
    # 인덱스 재 설정
    electrod_df = electrod_df.reset_index(drop=True)

    # 중복 값 제거
    electrod_df.drop_duplicates()
    
    return electrod_df

def generate_moving_average(data,size=30):
    data['Moving_Average'] = data['ELECTRODE_EFFICIENCY'].rolling(window=size).mean()
    data = data.dropna(subset=['Moving_Average'])
   
    return data


def load_model_from_pickle(file_path):
    """
    피클 파일에서 모델을 로드하는 함수.
    
    Parameters:
    - file_path: str, 피클 파일 경로
    
    Returns:
    - model: 피클 파일에서 로드된 모델
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"모델이 성공적으로 '{file_path}'에서 로드되었습니다.")
        return model
    except FileNotFoundError:
        print(f"에러: '{file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")


def save_model_from_pickle(model_name,model):
    with open(model_name, 'wb') as file:
        pickle.dump(model, file)
        
    print(f"모델이 {model_name}.pkl 파일로 저장되었습니다.")


def find_ship(ship_name):
    ship_df = ship_info[ship_info['ship_name']==ship_name] 
    ship_id = ship_df['ship_id']
    
    return ship_id.values


# In[8]:


def plot_time_series(data):
    # Plotting the line graph for ELECTRODE_EFFICIENCY

    # Calculating the moving average (window size of 5, can be adjusted)
    #data['Moving_Average'] = data['ELECTRODE_EFFICIENCY'].rolling(window=30).mean()

    plt.figure(figsize=(12, 6))
    #plt.plot(data['ELECTRODE_EFFICIENCY'], linestyle='-',label='Electrode Efficiency',color='orange')
    plt.plot(data['Moving_Average'], linestyle='--', color='red', label='Moving Average (5)')

    # 12000마다 수직선 그리기
    for i in range(0, len(data), 12000):
        plt.axvline(x=i, color='green', linestyle='--', linewidth=0.7)

    plt.title('Electrode Efficiency Over Time')
    plt.ylabel('Electrode Efficiency')
    plt.grid(True)
    plt.show()

    
def plot_time_series_with_time(data):
    # Plotting the line graph for ELECTRODE_EFFICIENCY

    plt.figure(figsize=(12, 6))
    plt.plot(data['DATA_TIME'], data['Moving_Average'], linestyle='--', color='green', label='Moving Average (5)')

    # 12000마다 수직선 그리기
    for i in range(0, len(data), 12000):
        plt.axvline(x=i, color='green', linestyle='--', linewidth=0.7)

    plt.title('Electrode Efficiency Over Time')
    plt.xticks(rotation=90)
    plt.ylabel('Electrode Efficiency')
    plt.grid(True)
    plt.show()    


def plot_trend_by_section(df):
    
    # 슬라이딩 윈도우 설정
    window_size = 1000  # 각 구간의 크기
    step_size = 100    # 윈도우 이동 간격
    results = []

    # 슬라이딩 윈도우를 사용하여 구간별 감소 여부 파악
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_data = df.iloc[start:end]
        x = range(window_size)  # 윈도우 내 인덱스
        y = window_data['ELECTRODE_EFFICIENCY']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # 감소 경향 확인
        is_decreasing = slope < 0  # 기울기가 음수이면 감소 경향
        results.append({
            'Start_Time': window_data['DATA_TIME'].iloc[0],
            'End_Time': window_data['DATA_TIME'].iloc[-1],
            'Slope': slope,
            'Is_Decreasing': is_decreasing
        })

    # 결과 데이터프레임 생성
    results_df = pd.DataFrame(results)

    # 감소 경향이 있는 구간 필터링
    decreasing_segments = results_df[results_df['Is_Decreasing']]

    # 감소 경향이 있는 구간 출력
    print("감소 경향이 있는 구간:")
    print(decreasing_segments)

    # 감소 경향이 있는 일부 구간 시각화
    plt.figure(figsize=(12, 6))
    for index, row in decreasing_segments.tail(5).iterrows():  # 첫 5개 구간만 시각화
        window_data = df[(df['DATA_TIME'] >= row['Start_Time']) & (df['DATA_TIME'] <= row['End_Time'])]
        plt.plot(window_data['DATA_TIME'], window_data['ELECTRODE_EFFICIENCY'], label=f"구간 {row['Start_Time']} - {row['End_Time']}")

    plt.title('구간별 감소 경향 분석 (일부 구간)')
    plt.xlabel('Time')
    plt.ylabel('Efficiency')
    plt.grid(True)
    plt.legend()
    plt.show()


def make_train_dataset(data, time_steps):
    # 예시 데이터프레임 생성 (실제 데이터로 대체하세요)
    # 여기서는 ELECTRODE_EFFICIENCY를 사용한다고 가정

    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Moving_Average']])

    # 시계열 데이터 준비 함수
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps, 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    # 시계열 길이 설정 및 데이터 분할

    X, y = create_sequences(scaled_data, time_steps)

    # 데이터 형태 조정 (LSTM input 형식)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 데이터셋 분할 (80% 학습, 20% 테스트)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train,X_test,y_train,y_test,scaler

def create_sequences_test(data, time_steps):
    # 데이터 스케일링
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Moving_Average']])
    
    X, y = [], []
    
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps, 0])
        y.append(scaled_data[i + time_steps, 0])
        
    X,y = np.array(X), np.array(y)
    
    # 데이터 형태 조정 (LSTM input 형식)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X,y,scaler


def train_lstm_obj(X_train,y_train, time_steps):

    # NNAR 모델 정의
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # 모델 학습
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    
    return model


def  plot_predict_values(model,X_test,y_train,y_test,scaler):

        # 예측 수행
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

        # 실제 값과 예측 값 비교

        # 실제 값 재조정 (Rescale)
        y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        #predicted_values_rescaled = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

        # 결과 시각화

        plt.figure(figsize=(14, 6))

        # 학습 데이터의 실제 값 시각화
        plt.plot(y_train_rescaled, label='Train Actual', color='blue')

        plt.plot(range(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)), y_test_rescaled, label='Test Actual')
        plt.plot(range(len(y_train_rescaled), len(y_train_rescaled) + len(y_test_rescaled)), predictions, label='Predicted', linestyle='--')
        #plt.plot(range(len(y_train_rescaled) + len(y_test_rescaled), len(y_train_rescaled) + len(y_test_rescaled)+2000),predicted_values_rescaled,label='Predicted Over', linestyle='--')

        plt.title('Lstm Model Predictions')
        plt.xlabel('Time Steps')
        plt.ylabel('Efficiency') 
        plt.legend()
        plt.show()
        
        return y_test_rescaled, predictions


def calculate_evaluation(y_test_rescaled,predictions):
    # 평가 지표 계산

    mse = mean_squared_error(y_test_rescaled, predictions)
    rmse = np.sqrt(mse)

    # 결과 출력
    print(f"Root Mean Squared Error (RMSE): {rmse}")


def make_step_by_step_prediction(scaled_data,model,time_steps):

    # 단계별 예측을 통한 6000분 후 예측
    # 초기 입력 데이터로 마지막 10분 데이터 사용
    # 설정
    # 동안의 데이터를 사용하여 예측

    predicted_values = []
    current_sequence = scaled_data[-time_steps:]  # 마지막 time_steps 분간의 데이터

    for _ in range(time_steps):
        # LSTM이 요구하는 입력 형태로 변환
        current_sequence_reshaped = current_sequence.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)

        # 다음 값을 예측
        predicted_value = model.predict(current_sequence_reshaped)[0, 0]
        predicted_values.append(predicted_value)

        # 예측된 값을 현재 시퀀스에 추가하고, 맨 앞의 값을 제거
        current_sequence = np.append(current_sequence[1:], predicted_value)

    # 2000분 후 예측된 마지막 값 출력
    #print(f"200분 후 예측된 값: {predicted_values[-1]}")
    
    return predicted_values



def plot_seasonal(data):
    """ Seasonal Decomposition (Optional if you suspect seasonal pattern)
    """
    
    result = seasonal_decompose(data['Moving_Average'].dropna(), model='additive', period=1000)  # Adjust period as needed
    result.plot()
    plt.show()

    

def make_step_by_step_prediction(scaled_data,model,turns):
    """ 단계별 예측 함수
    """
    

    # 단계별 예측을 통한 6000분 후 예측
    # 초기 입력 데이터로 마지막 10분 데이터 사용
    # 설정
    # 동안의 데이터를 사용하여 예측

    predicted_values = []
    current_sequence = scaled_data[-1:]  # 마지막 time_steps 분간의 데이터

    for i in range(turns):
    
        # LSTM 모델은 (배치 크기, 타임스텝, 피처 수) 형태의 데이터를 입력받아야 하므로,
        # current_sequence의 i번째 타임스텝을 모델에 입력 (즉, (1, 100, 1) 형태로 변환)
        input_sequence = current_sequence[i].reshape((1,scaled_data.shape[1],1))  # (1, 100, 1)
        
        # 다음 값을 예측
        predicted_value = model.predict(input_sequence)
        
        # 예측된 값을 predicted_values에 추가
        predicted_values.append(predicted_value.item())

        # 예측된 값을 현재 시퀀스에 추가하고, 맨 앞의 값을 제거
        predicted_value_reshaped = np.array(predicted_value).reshape(1, 1)  # (1, 1)으로 변환
        
         # 마지막 시퀀스의 첫 번째 값을 제거하고, 예측된 값을 끝에 추가
        last_sequence = current_sequence[-1]  # 마지막 시퀀스 (크기: (100, 1))
        
        # new_value를 (1, 100, 1) 형식으로 변환
        new_value_reshaped = np.append(last_sequence[1:], predicted_value).reshape(1,scaled_data.shape[1],1)
        
        # current_sequence의 끝에 new_value_reshaped를 추가 (concatenate 사용)
        current_sequence = np.concatenate((current_sequence, new_value_reshaped), axis=0)

        
    # 2000분 후 예측된 마지막 값 출력
    
    return current_sequence, predicted_values
