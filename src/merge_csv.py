from pathlib import Path
import pandas as pd
import itertools
from argparse import ArgumentParser
import os
import glob
import re


def merge_timestamp_csv(input_files: str, dataset_ID: str):

    input_dir_path = Path(input_files)
    
    list_audio = list(input_dir_path.glob(f'thalassa_{dataset_ID}_[0-9]*.csv'))
    
    list_conca_dataset = []
    list_conca_filename = []
    list_conca_start_time = []
    list_conca_end_time = []
    list_conca_start_freq = []
    list_conca_end_freq = []
    list_conca_annotation = []
    list_conca_annotator = []
    list_conca_start_dt = []
    list_conca_end_dt = []
    list_conca_is_box = []
    
    for ll in list_audio:
    
        print(f"read and remove file {ll}")
        list_conca_dataset.append(list(pd.read_csv(ll)['dataset'].values))
        list_conca_filename.append(list(pd.read_csv(ll)['filename'].values))
        list_conca_start_time.append(list(pd.read_csv(ll)['start_time'].values))
        list_conca_end_time.append(list(pd.read_csv(ll)['end_time'].values))
        list_conca_start_freq.append(list(pd.read_csv(ll)['start_frequency'].values))
        list_conca_end_freq.append(list(pd.read_csv(ll)['end_frequency'].values))
        list_conca_annotation.append(list(pd.read_csv(ll)['annotation'].values))
        list_conca_annotator.append(list(pd.read_csv(ll)['annotator'].values))
        list_conca_start_dt.append(list(pd.read_csv(ll)['start_datetime'].values))
        list_conca_end_dt.append(list(pd.read_csv(ll)['end_datetime'].values))
        list_conca_is_box.append(list(pd.read_csv(ll)['is_box'].values))
        os.remove(ll)
            
    df = pd.DataFrame({"dataset": list(itertools.chain(*list_conca_dataset)),\
                       "filename": list(itertools.chain(*list_conca_filename)), \
                       "start_time": list(itertools.chain(*list_conca_start_time)), \
                       "end_time": list(itertools.chain(*list_conca_end_time)), \
                       "start_frequency": list(itertools.chain(*list_conca_start_freq)), \
                       "end_frequency": list(itertools.chain(*list_conca_end_freq)), \
                       "annotation": list(itertools.chain(*list_conca_annotation)), \
                       "annotator": list(itertools.chain(*list_conca_annotator)), \
                       "start_datetime": list(itertools.chain(*list_conca_start_dt)), \
                       "end_datetime": list(itertools.chain(*list_conca_end_dt)), \
                       "is_box": list(itertools.chain(*list_conca_is_box))})

    df.sort_values(by=["start_datetime"], inplace=True)
    
    df_name = 'thalassa_' + dataset_ID + '.csv'
    df_path = input_dir_path.joinpath(df_name)
    
    if os.path.exists(df_path):
        os.remove(df_path)
    
    df.to_csv(
        df_path,
        index=False
    )
    
    print(f"save file {str(df_path)}")

if __name__ == "__main__":
    parser = ArgumentParser()
    required = parser.add_argument_group("required arguments")
    
    required.add_argument(
        "--input-files",
        "-i",
        help="The csv to be reshaped, as either the path to a directory containing audio files and a timestamp.csv or a list of filenames all in the same directory alongside a timestamp.csv.",
    )
    
    # Adding a new argument for dataset_ID
    required.add_argument(
        "--dataset-ID",
        "-n",
        help="The dataset ID to be passed to merge_timestamp_csv function.",
    )

    args = parser.parse_args()

    input_files = (
        args.input_files.split(" ")
        if not Path(args.input_files).is_dir()
        else args.input_files
    )
    
    files = merge_timestamp_csv(input_files=input_files, dataset_ID=args.dataset_ID)
