import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BWMSVisualizer:
    def __init__(
            self, 
            base_data: pd.DataFrame, 
            combined_data: pd.DataFrame,
            ship_id: str,
            op_index: int,
            section: int,
            op_type: int
            ) -> None: 
        self.base_path = 'D:\\bwms\\preprocessing'
        self.outlier_ballast_dict = {'CSU': [0, 49.46], 'STS': [0, 33.29], 'FTS': [0, 39.24], 
                                     'FMU': [286, 2933], 'TRO': [0, 8], 'CURRENT': [0, 18790], 'VOLTAGE': [3.0, 4.7]}
        self.outlier_deballast_dict = {'CSU': [0, 49.46], 'STS': [0, 33.29], 'FMU': [286, 2933], 
                                       'TRO': [0, 1.79], 'ANU': [0, 1320]}
        self.base_data = base_data
        self.combined_data = combined_data
        self.ship_id = ship_id
        self.op_index = op_index
        self.section = section
        self.op_type = op_type
        sns.set(style="whitegrid")  # Seaborn 스타일 설정

    def find_folder(self, file_path: str) -> None:
        """Ensure the directory for the file path exists."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_plot(self, file_path: str) -> None:
        """Save and close the current plot."""
        self.find_folder(file_path)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close('all')

    def plot_histograms_with_noise(self) -> None:
        """Plot histograms with noise detection for ballasting and deballasting operations."""
        fig = plt.figure(figsize=(15, 8))
        op_type = self.combined_data['OP_TYPE'].iloc[0]
        columns = self.outlier_ballast_dict.keys() if op_type != 2 else self.outlier_deballast_dict.keys()
        outlier_dict = self.outlier_ballast_dict if op_type != 2 else self.outlier_deballast_dict
        palette = ['#2ca02c', '#1f77b4']

        for i, column in enumerate(columns, start=1):
            plt.subplot(2, (len(columns) + 1) // 2, i)
            sns.kdeplot(data=self.combined_data[self.combined_data['state'] == 'original'], x=column, fill=True, label='original', color=palette[0])
            sns.kdeplot(data=self.combined_data[self.combined_data['state'] == 'preprocessing'], x=column, fill=True, label='preprocessed', color=palette[1])
            plt.axvline(outlier_dict[column][0], color='orange', linestyle='--', alpha=0.8)
            plt.axvline(outlier_dict[column][1], color='orange', linestyle='--', alpha=0.8)

        plt.tight_layout()
        file_path = os.path.join(self.base_path, f"{self.ship_id}/{self.op_index}/{self.ship_id}_{self.op_index}_{self.section}_noise.png")
        self.save_plot(file_path)

    def plot_bar_with_operation(self) -> None:
        """Plot bar chart for operation data before and after preprocessing."""
        total_value = self.base_data['DATA_INDEX'].count()
        top_60_value = self.base_data['DATA_INDEX'][self.base_data['DATA_INDEX'] <= 60].count()
        total_index_after = total_value - top_60_value

        plt.figure(figsize=(8, 6))
        x_positions = [0, 1]
        plt.bar(x_positions[0], total_value, label='Total (All DATA_INDEX)', color='#7F9DB9', width=0.4, alpha=0.6)
        plt.bar(x_positions[0], top_60_value, label='Top 60 DATA_INDEX', color='#A0A3A4', width=0.4, alpha=0.6)
        plt.bar(x_positions[0], top_60_value, color='None', edgecolor='#BD7677', hatch='//', width=0.4, alpha=0.6)
        plt.text(x=x_positions[0], y=top_60_value/2, s='Operation Data to be deleted', fontsize=10, color='white', ha='center')

        plt.bar(x_positions[1], total_index_after, label='Remaining (After preprocessing)', color='#BD7677', width=0.4, alpha=0.7)
        plt.axhline(y=total_index_after, color='red', linestyle='-', linewidth=1, alpha=0.7)
        plt.text(x=x_positions[1], y=total_index_after+5, s=f"After preprocessing: {total_index_after}", color='#BD7677', fontsize=12, ha='center')
        plt.text(x=x_positions[1], y=total_index_after/2, s='Deleted Data Remaining', color='white', fontsize=10, ha='center')

        plt.gca().set_facecolor('#F7F7F7')
        plt.xticks(x_positions, ['Original Data', 'After Preprocessing'])
        plt.ylabel('Values', fontsize=12)
        plt.tight_layout()

        file_path = os.path.join(self.base_path, f"{self.ship_id}/{self.op_index}/{self.ship_id}_{self.op_index}_{self.section}_operation.png")
        self.save_plot(file_path)

    def plot_pie_with_duplication(self) -> None:
        """Plot pie charts showing duplication before and after preprocessing."""
        duplicates = self.base_data[self.base_data.duplicated(subset=['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME'], keep='first')]
        non_duplicates = self.base_data[~self.base_data.duplicated(subset=['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_TIME'], keep='first')]

        before_ratio = len(duplicates)
        after_ratio = len(self.base_data) - before_ratio

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.pie(
            [before_ratio, after_ratio], 
            labels=['Duplication', 'Available'], 
            autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * (before_ratio + len(self.original_data) - before_ratio) / 100),  
            colors=['#A020F0', '#40E0D0']
            )
        plt.title('Before Removal')

        plt.subplot(1, 2, 2)
        plt.pie(
            [after_ratio, len(self.base_data) - after_ratio], 
            labels=['Available', 'Duplication'], 
            autopct=lambda p: '{:.1f}% ({:.0f})'.format(p, p * (after_ratio + len(non_duplicates) - after_ratio) / 100), 
            colors=['#40E0D0', '#A020F0']
            )
        plt.title('After Removal')

        plt.tight_layout()
        file_path = os.path.join(self.base_path, f"{self.ship_id}/{self.op_index}/{self.ship_id}_{self.op_index}_{self.section}_duplication.png")
        self.save_plot(file_path)

    def plot_double_bar_with_missing(self) -> None:
        """Plot double bar charts comparing counts with and without missing data."""
        total_counts = self.base_data.count()
        missing_counts = self.base_data.isnull().sum()
        adjusted_counts = total_counts - missing_counts

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(total_counts.index, total_counts, color='#3A9DBF', label='Total Count (No Missing)')
        ax.barh(total_counts.index, missing_counts, color='#D3D3D3', label='Missing Count')
        ax.barh(total_counts.index, adjusted_counts, left= -adjusted_counts, color='#6A1B9A', label='Total Count (After removing the Missing )', align='center', alpha=0.8)

        ax.axvline(x=0, color='black', linewidth=0.5)
        current_ticks = ax.get_xticks()
        ax.set_xticklabels([str(abs(int(tick))) for tick in current_ticks])
        ax.set_xlabel('Count')
        ax.set_ylabel('Variables')
        ax.set_title('Comparison of Counts with and without Missing Data')
        ax.legend()

        plt.tight_layout()
        file_path = os.path.join(self.base_path, f"{self.ship_id}/{self.op_index}/{self.ship_id}_{self.op_index}_{self.section}_missing.png")
        self.save_plot(file_path)
