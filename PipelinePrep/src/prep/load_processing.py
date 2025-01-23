# basic
import json
import pandas as pd
import os
from typing import Tuple, Optional

# module
from prep.preprocessing import PreprocessingPipeline
from prep_visualizer.prep_visualize import BWMSVisualizer

# module.dataline
from prep_dataline  import get_dataframe_from_database
from prep_dataline import load_database
from stat_dataline import logger
from prep.preprocessing import PreprocessingPipeline

class DataPreprocessor:
    def __init__(self, db_target: str, db_source: Tuple[str, str], base_path: str):
        self.db_source = db_source
        self.db_target = db_target
        self.base_path = base_path

    def find_folder(self, file_path: str):
        """Ensure the directory exists for a given file path."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def process_preprocessed_data(self, indicator_data: dict, preprocessed_data: pd.DataFrame):
        """Process and save preprocessed data."""
        dict_dataframe = pd.DataFrame([preprocessed_data])
        concat = pd.concat([dict_dataframe, indicator_data], axis=1)
        print(f"[INFO] concat: \n {concat.columns}")
        concat = concat[['SHIP_ID', 'OP_INDEX', 'SECTION', 'OP_TYPE', 'NOISE', 'MISSING',
                         'DUPLICATE', 'OPERATION', 'DATA_COUNT', 'PRE_COUNT',
                         'START_DATE', 'END_DATE', 'REG_DATE']]
        self.load_database('test_tc_data_preprocessing_flag', concat)

    def configure_columns(self):
        self.df = self.df[
            [
                'SHIP_ID','OP_INDEX','SECTION','DATA_TIME','DATA_INDEX',
                'CSU','STS','FTS','FMU','TRO','ANU','RATE','CURRENT','VOLTAGE'
            ]
        ]

        self.optime = self.optime[
            [
                'SHIP_ID','OP_INDEX','OP_TYPE','START_TIME','END_TIME','RUNNING_TIME'
            ]
        ]

    def distribute_by_application(self, ship_id: str, op_index: str, section: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Process data by application."""
        self.df = get_dataframe_from_database(self.db_source[0], ship_id=ship_id, op_index=op_index, section=section)
        self.optime = get_dataframe_from_database(self.db_source[1], optime=True, ship_id=ship_id, op_index=op_index)
        self.configure_columns()

        if self.optime.empty:
            logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=op_time dataframe is empty| TYPE=preprocessing | IS_PROCESSED=False')
            print("self.optime DataFrame 비어있습니다.")
            return None, None

        op_type = self.optime.iloc[0]['OP_TYPE']
        date_time = self.optime.iloc[0]['START_TIME']

        if op_type not in [2, 4]: # Ballast
            sensor = pd.merge(self.optime, self.df, on=['SHIP_ID', 'OP_INDEX'], how='left')
            preprocessor_pipeline = PreprocessingPipeline(sensor)
            results = preprocessor_pipeline.apply_pipeline()
            original_data = results.get('original_data')
            processed_data = results.get('processed_data')

            if processed_data is None:
                return original_data, None
            
            self.process_preprocessed_data(results.get('count_df'), results.get('text_dict'))
            file_path = os.path.join(self.base_path, f"{ship_id}/{op_index}/{ship_id}_{op_index}_{section}_file_ba.json")
            self.find_folder(file_path)

            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(results['text_dict'], json_file, ensure_ascii=False, indent=4)

            original_data['state'] = 'original'
            processed_data['state'] = 'preprocessing'
            concat_data = pd.concat([original_data, processed_data])

            bwms_visualizer = BWMSVisualizer(original_data,concat_data,ship_id, op_index, section, op_type)
            bwms_visualizer.plot_histograms_with_noise()
            bwms_visualizer.plot_bar_with_operation()
            bwms_visualizer.plot_pie_with_duplication()
            bwms_visualizer.plot_double_bar_with_missing()
            result_data = processed_data[['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_INDEX']]
            self.load_database(self.db_target, 'test_tc_data_preprocessing_result_flag', result_data)

            return original_data, processed_data
            # try:
            #     preprocessor_pipeline = PreprocessingPipeline(sensor)
            #     results = preprocessor_pipeline.apply_pipeline()
            #     original_data = results.get('original_data')
            #     processed_data = results.get('processed_data')

            #     if processed_data is None:
            #         return original_data, None
                
            #     self.process_preprocessed_data(results.get('count_df'), results.get('text_dict'))
            #     file_path = os.path.join(self.base_path, f"{ship_id}/{op_index}/{ship_id}_{op_index}_{section}_file_ba.json")
            #     self.find_folder(file_path)

            #     with open(file_path, 'w', encoding='utf-8') as json_file:
            #         json.dump(results['text_dict'], json_file, ensure_ascii=False, indent=4)

            #     original_data['state'] = 'original'
            #     processed_data['state'] = 'preprocessing'
            #     concat_data = pd.concat([original_data, processed_data])

            #     bwms_visualizer = BWMSVisualizer(original_data,concat_data,ship_id, op_index, section, op_type)
            #     bwms_visualizer.plot_histograms_with_noise()
            #     bwms_visualizer.plot_bar_with_operation()
            #     bwms_visualizer.plot_pie_with_duplication()
            #     bwms_visualizer.plot_double_bar_with_missing()
            #     result_data = processed_data[['SHIP_ID', 'OP_INDEX', 'SECTION', 'DATA_INDEX']]
            #     self.load_database(self.db_target, 'test_tc_data_preprocessing_result_flag', result_data)

            #     return original_data, processed_data

            # except ValueError as e:
            #     print(f"Error: {e}. Skipping to the next iteration.")
            #     return original_data, None

        else: # Deballast
            logger.info(f'SHIP_ID={ship_id} | OP_INDEX={op_index} | SECTION={section} | START_TIME={date_time} | LOG_ENTRY=The op_type is deballast | TYPE=preprocessing | IS_PROCESSED=False')
            return None, None