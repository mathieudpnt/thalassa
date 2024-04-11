# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from scipy import signal
from csv import writer
import glob
import soundfile as sf
import sys

from utils import click_train_extraction, extract_datetime, gabor


# imported parameters
batch_id = int(sys.argv[1]) + 1
path_osmose_dataset = sys.argv[2]

if sys.argv[3] == 'False':
    campaign_ID = ''
else:
    campaign_ID = sys.argv[3]

dataset_ID = sys.argv[4]
dataset_resolution = sys.argv[5]
UTC_offset = sys.argv[6]
all_files = int(sys.argv[7])
batch_size = int(sys.argv[8])

params= {}
params['model'] = int(sys.argv[9])
params['threshold'] = int(sys.argv[10])
params['ICI_max'] = float(sys.argv[11])
params['sum_peaks'] = int(sys.argv[12])
params['h_morph'] = int(sys.argv[13])
params['l_morph'] = int(sys.argv[14])
params['width_min'] = int(sys.argv[15])
params['f_min'] = int(sys.argv[16])
params['f_max'] = int(sys.argv[17])

if sys.argv[18] == 'True':
    params['output_image'] = True
else:
    params['output_image'] = False

wav_files = sys.argv[19:]

dataset_path = os.path.join(path_osmose_dataset, os.path.join(campaign_ID, dataset_ID))
data_path = os.path.join(dataset_path, 'data', 'audio', dataset_resolution)
batch_num = (int(all_files) + batch_size - 1) // batch_size

print(f"\n%% batch {batch_id}/{batch_num}\n")
print(f"path_osmose_dataset : {path_osmose_dataset}")
print(f"campaign_ID : {campaign_ID}")
print(f"dataset_ID : {dataset_ID}")
print(f"dataset_resolution : {dataset_resolution}")
print(f"UTC_offset : {UTC_offset}")
print(f"number of files : {len(wav_files)}/{int(all_files)}")
print(f"first file : {wav_files[0]}")
print(f"last file : {wav_files[-1]}")

# %% output folder
output_path = os.path.join(dataset_path, 'processed', 'thalassa')
print(f'output folder: {output_path}')
spectro_path = os.path.join(output_path, 'spectro')

if not os.path.exists(os.path.join(dataset_path, 'processed')):
    os.mkdir(os.path.join(dataset_path, 'processed'))
if not os.path.exists(output_path):
    os.mkdir(output_path)

if params['output_image'] is True:
    print(f'spectro folder: {spectro_path}')
# %% create csv detections
name_csv = 'thalassa_' + dataset_ID + '_' + str(batch_id) +'.csv'
path_csv = os.path.join(output_path, name_csv)

header = ['dataset','filename','start_time','end_time','start_frequency','end_frequency','annotation','annotator','start_datetime','end_datetime','is_box']
with open(path_csv, 'w+', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(header)
    f_object.close()
    
print(f'detection file: {path_csv}')

# %% BOUCLE DE TRAITEMENT
# Main loop

start_time = pd.Timestamp.now()
nb_WD, nb_SD = 0, 0

for file in wav_files:

    sig, Fe = sf.read(file)

    if Fe < 40e3:
        raise ValueError(f'Sampling frequency not high enough for click detection: {Fe} Hz')

    # in case of multichannel signal
    if len(np.shape(sig)) > 1:
        sig = sig[:, 0]

    duration = len(sig) / Fe

    f_min = params['f_min']
    f_max = min(Fe / 2, params['f_max'])

    # Gabor wave model used for crossed-correlation
    gabor_model = gabor(model=params['model'], N=len(sig), dur=duration)

    # Cross-correlation between the signal and the chosen model
    inter = signal.correlate(sig, gabor_model, mode='same')

    # Train click extraction
    click_train = click_train_extraction(data=inter,
                                         Fe=Fe,
                                         f_min=f_min,
                                         f_max=f_max,
                                         ratio=50000,
                                         energy_threshold=params['threshold'],
                                         ICI_max=params['ICI_max'],
                                         nb_click_min=params['sum_peaks'],
                                         height_morph=params['h_morph'],
                                         width_morph=params['l_morph'],
                                         width_min=params['width_min'])

    # Save detected click train
    if click_train != []:

        # Datetime extraction
        date_format = extract_datetime(file)

        # Add weak detection (WD) line to csv
        dur_spectro = int(dataset_resolution.split('_')[0])

        WD_beg = (date_format).strftime('%Y-%m-%dT%H:%M:%S.000') + UTC_offset + ':00'
        WD_end = (date_format + pd.Timedelta(seconds=dur_spectro)).strftime('%Y-%m-%dT%H:%M:%S.000') + UTC_offset + ':00'
        WD = [dataset_ID, os.path.basename(file), 0, dur_spectro, 0, f_max, 'Odontocete click detection', 'thalassa', WD_beg, WD_end, 0]
        nb_WD += 1
        with open(path_csv, 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(WD)
            f_object.close()

        # Add strong detection lines to csv (box)
        for train in click_train:
            SD_beg = (date_format + pd.Timedelta(seconds=train[0])).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + UTC_offset + ':00'
            SD_end = (date_format + pd.Timedelta(seconds=train[1])).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + UTC_offset + ':00'
            SD = [dataset_ID, os.path.basename(file), train[0], train[1], f_min, f_max, 'Odontocete click detection', 'thalassa', SD_beg, SD_end, 1]
            nb_SD += 1
            with open(path_csv, 'a', newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(SD)

            # Save annotated spectrogram if asked
            if params['output_image'] is True:
                # Spectrogram computation
                overlap = 0.6
                nperseg = 512
                f, t, Sxx = signal.spectrogram(sig, Fe, nperseg=nperseg, window=('hamming'), noverlap=round(overlap * nperseg))

                # Spectrogram plot
                plt.imshow(20 * np.log10(abs(Sxx)), aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]], cmap=cm.viridis)
                plt.xlim([0, int(dataset_resolution.split('_')[0])])
                plt.xticks([], [])
                plt.yticks([], [])
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # delete white borders

                # Display strong detections
                for train in click_train:
                    width = train[-1] - train[0]
                    height = f_max - f_min
                    plt.gca().add_patch(Rectangle((train[0], f_min), width, height, fc='none', ec='k', lw=1))

                # Save plot
                plt.savefig(os.path.join(spectro_path, os.path.basename(file).split('wav')[0] + 'png'), dpi=150)
                plt.close()

end_time = pd.Timestamp.now()

print(f'Time taken: {end_time - start_time}')
print(f'Weak detections: {nb_WD}')
print(f'Strong detections: {nb_SD}')

# %% Configuration file
if batch_id == batch_num:

    txt_path = os.path.join(output_path, 'thalassa_' + dataset_ID + '_config' + '.txt')
    dt_str = pd.Timestamp.now().strftime('%d-%m-%yT%H-%M-%S')

    with open(txt_path, "w+") as file:
        file.write('%%% PARAMETERS %%%\n\n')
        file.write(f'Creation of the parameter file: {dt_str}\n')

        for var_name, var_value in params.items():
            file.write(f"{var_name}: {var_value}\n")

        file.write(f'dataset path: {dataset_path}\n')
        file.write(f'data path: {data_path}\n')
        file.write(f'dataset resolution: {dataset_resolution}\n')
        file.write(f'detection file: {path_csv}\n')

    print(f"configuration file: {txt_path}")
