from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseHunting(ABC):
    def label_hunting_multiple_of_two(self, data) -> pd.DataFrame:
        """
        공통 로직: 변화율 계산, 임계값 라벨링, 헌팅 라벨링
        """
        # 공통 로직 1: 변화율 계산
        data = self.calculate_rate_of_change(data)

        # 공통 로직 2: 임계값 계산 및 라벨링
        data = self.extract_rate_of_change_threshold(data)

        # 공통 로직 3: 헌팅 라벨링
        data = self.handle_peaks_and_valleys(data)

        # 자식 클래스에서 추가 로직을 수행할 수 있도록 제공
        data = self.add_custom_logic(data)
        
        return data

    @abstractmethod
    def calculate_rate_of_change(self, df):
        """
        [다형성] 변화율 계산 메서드: 자식 클래스에서 구현
        """

        pass 
    
    def extract_rate_of_change_threshold(self, data):
        """
        임계값 계산 및 라벨링: 공통 로직
        """
        
        positive_threshold = data['TRO_Ratio'].quantile(0.95)
        negative_threshold = data['TRO_Ratio'].quantile(0.05)
        data['Peak'] = (data['TRO_Ratio'] > positive_threshold) & (data['TRO_Ratio'] >= 2)
        data['Valley'] = (data['TRO_Ratio'] < negative_threshold) & (data['TRO_Ratio'] <= -2)
        return data
    

    def handle_peaks_and_valleys(self, df):
        """
        헌팅 라벨링: 피크와 밸리의 번갈아 나타나는 패턴 확인
        """

        df['HUNTING'] = 0
        peaks_indices = df[df['Peak']].index
        valleys_indices = df[df['Valley']].index

        sequence = []
        for index in np.sort(np.concatenate((peaks_indices, valleys_indices))):
            if not sequence:
                sequence.append(index)
            elif (sequence[-1] in peaks_indices and index in valleys_indices) or \
                 (sequence[-1] in valleys_indices and index in peaks_indices):
                sequence.append(index)
            else:
                continue

            if len(sequence) >= 6 and len(sequence) % 3 == 0:
                df.at[sequence[-1], 'HUNTING'] = 1
                sequence = []

        return df
    

    def add_custom_logic(self, df):
        """
        추가 로직: 자식 클래스에서 필요 시 구현
        """
        return df
