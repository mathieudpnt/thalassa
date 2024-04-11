# %%
import os
import numpy as np
import time
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from datetime import timedelta
import datetime as dt
from scipy.io import wavfile
from scipy.signal import correlate
from csv import writer
from tqdm import tqdm
import glob
import shutil
import pandas as pd

from utils import bloc_traitement, extraction, calcul_spectro, extraction_dt_format, gabor_1, gabor_2, gabor_3

# %%
# imported parameters
batch_id = 0
path_osmose_dataset = '/home/datawork-osmose/dataset/'
dataset_ID = 'APOCADO_C2D1_IROISE_07072022'
dataset_resolution = '10_144000'
UTC_offset = '+02'
batch_size = 50

dataset_path = os.path.join(path_osmose_dataset, dataset_ID)
data_path = os.path.join(dataset_path, 'data', 'audio', dataset_resolution)
wav_files = sorted(glob.glob(os.path.join(data_path, '**/*.wav'), recursive=True))[0:20]

# %%
# create csv detections
name_csv = 'thalassa_' + dataset_ID + '_' + str(batch_id) +'.csv'
path_csv = os.path.join(output_path, name_csv)

header = ['dataset','filename','start_time','end_time','start_frequency','end_frequency','annotation','annotator','start_datetime','end_datetime','is_box']
with open(path_csv, 'w+', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(header)
    f_object.close()
    
print(f'detection file: {path_csv}')

# %%
params = {"dataset_ID": 'test_pool',
          "model": 3,
          "threshold": 85,
          "ICI_max": 0.25,
          "sum_peaks": 8,
          "h_morph": 115,
          "l_morph": 1,
          "width_min": 6,
          "f_min": 20000,
          "f_max": 130000,
          "UTC_offset": '+02',
          "output_image": False,
          "path_csv": path_csv,
          "dataset_resolution": dataset_resolution
          }


# %%
def process_thalassa(file, params):

    # Récupération des caractéristiques du fichier audio :
    # durée totale ; fréquence d'échantillonnage ; longueur vectorielle
    Fe, sig = wavfile.read(file)
    N = len(sig)
    t_tot = round(N/Fe)

    # Définition liminaire de la plage d'observation fréquentielle
    # NB : la condition « y == 0 » permet de ne le calculer
    # qu'une seule fois, lors du traitement du premier fichier
    f_min = params['f_min']
    f_max = min(Fe / 2, params['f_max'])

    # Pour que l'algorithme puisse fonctionner si le signal de l'hydrophone
    # est enregistré en stéréo, on pose une condition pour ne conserver que
    # le premier canal du signal :
    if len(np.shape(sig)) > 1:
        sig = sig[:, 0]

    # Calcul du modèle de Gabor choisi pour l'intercorrélation
    # NB : la condition « y == 0 » permet de ne le calculer
    # qu'une seule fois, lors du traitement du premier fichier
        
    if params['model'] == 1:
        modele = gabor_1(N, t_tot)
    if params['model'] == 2:
        modele = gabor_2(N, t_tot)
    if params['model'] == 3:
        modele = gabor_3(N, t_tot)
    else:
        raise ValueError(f'model not well defined')

    # Intercorrélation du signal d'origine par le modèle précédemment calculé
    inter = correlate(sig, modele, mode='same')

    # Récupération de la liste contenant les coordonnées des trains de clics
    L, l, f, t, liste_vert = bloc_traitement(inter, N, Fe, f_min, f_max, valeur_max=50000, pourcentage_seuil_energetique=params['threshold'], ICI_max=params['ICI_max'], nb_clics_min=params['sum_peaks'], h_morph=params['h_morph'], l_morph=params['l_morph'], larg_min=params['width_min'])

    # On ignore la suite de la boucle si L est vide, c'est-à-dire si
    # aucun train de clics n'est détecté. On passe immédiatement au
    # fichier audio suivant.
    if L == []:
        return
    else:
        # Si L n'est pas vide, on récupère donc les intervalles
        # temporels contenant les trains
        extraits, _ = extraction(L, N, t_tot, sig, l, params['ICI_max'])
        N_ex = len(extraits)
        
        # date extraction
        f, date_format = extraction_dt_format(file)
        match = re.search(f, file)
        if match:
            date_str = match.group()
            
        # add weak detection line to csv
        s_dbt = 0
        s_fin = int(params['dataset_resolution'].split('_')[0])
        
        res_dbt = dt.datetime.strptime(date_str, date_format) + timedelta(seconds=s_dbt)
        res_dbt = res_dbt.strftime('%Y_%m_%dT%H:%M:%S.000')
        res_fin = dt.datetime.strptime(date_str, date_format) + timedelta(seconds=s_fin)
        res_fin = res_fin.strftime('%Y_%m_%dT%H:%M:%S.000')

        D_dbt = res_dbt + params['UTC_offset'] + ':00'
        D_fin = res_fin + params['UTC_offset'] + ':00'
        
        liste_resultats = [params['dataset_ID'], os.path.basename(file), 0, 10, 0, f_max, 'Odontocete click detection', 'thalassa', D_dbt, D_fin, 0]
        
        with open(params['path_csv'], 'a',newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(liste_resultats)
            f_object.close()
        
        
        # add box detection line to csv
        for p in range(N_ex):

            s_dbt = extraits[p][0][0]
            s_fin = extraits[p][0][-1]
            res_dbt = dt.datetime.strptime(date_str, date_format) + timedelta(seconds=s_dbt)
            res_dbt = res_dbt.strftime('%Y_%m_%dT%H:%M:%S.%f')[:-3]            
            res_fin = dt.datetime.strptime(date_str, date_format) + timedelta(seconds=s_fin)
            res_fin = res_fin.strftime('%Y_%m_%dT%H:%M:%S.%f')[:-3]            
            
            D_dbt = res_dbt + params['UTC_offset'] + ':00'
            D_fin = res_fin + params['UTC_offset'] + ':00'

            liste_resultats = [params['dataset_ID'], os.path.basename(file), extraits[p][0][0], extraits[p][0][-1], liste_vert[p][1], liste_vert[p][0], 'Odontocete click detection', 'thalassa', D_dbt, D_fin, 1]
            
            with open(params['path_csv'], 'a',newline='') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(liste_resultats)
                f_object.close()

            if params['output_image'] is True:
                f, t, Sxx_sig = calcul_spectro(sig, Fe, 512, 0.6)
                plt.pcolormesh(t, f, 20 * np.log10(abs(Sxx_sig)), shading='Gouraud', cmap=cm.viridis)
                plt.xlim([0, 10])
                plt.xticks([], [])
                plt.yticks([], [])
                # affichage de la boîte contenant le train de clics, sur le spectrogramme
                # COIN EN BAS à gauche, puis largeur, puis hauteur
                for p in range(N_ex):
                    largeur = extraits[p][0][-1]-extraits[p][0][0]
                    hauteur = liste_vert[p][1]-liste_vert[p][0]
                    plt.gca().add_patch(Rectangle((extraits[p][0][0], liste_vert[p][0]), largeur, hauteur, fc='none', ec='k', lw=1))

                plt.savefig(output_path + file.split('wav')[0] + 'png', dpi=300)
                plt.close()
        return 1

# %%
total_time = dt.timedelta(seconds=0)

print(end_time-start_time)
print(total_time)
print(total_time + (end_time-start_time))

# %%
# Store current time before execution
start_time = dt.datetime.now()
pos = 0

# Run for loop
for file in tqdm(wav_files):
    out = process_thalassa(file, params)
    if out == 1:
        pos +=1

# Store current time after execution
end_time = dt.datetime.now()

# Print for loop execution time
print(f'Time taken for batch {batch_id}: ', end_time-start_time)

print('Files positive to detection: ', pos)
if pos == 0:
    os.remove(path_csv)
    print('Detection file was deleted as no detection has been made')

# %%

# %%
import os
from multiprocessing import Pool


# Store current time before execution
start_time = dt.datetime.now()

if __name__ == '__main__':
    # Create a pool to use all cpus
    pool = Pool(processes=12)
    pool.map(process_thalassa, wav_files)
    # Close the process pool
    pool.close()        

# Store current time after execution
end_time = dt.datetime.now()

# Print multi threading execution time
print('Time taken: ', end_time-start_time)

# %%
import os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Store current time before execution
start_time = dt.datetime.now()

if __name__ == '__main__':

    # Create a partially-applied function with constant_arg
    process_thalassa_partial = partial(process_thalassa, params=params)

    # Create a pool to use several CPUs
    with Pool(processes=4) as pool:
        # Use tqdm with Pool.imap_unordered for progress updates
        for _ in tqdm(pool.imap_unordered(process_thalassa_partial, wav_files), total=len(wav_files)):
            pass

# Store current time after execution
end_time = dt.datetime.now()

# Print multi-threading execution time
print('Time taken: ', end_time - start_time)


# %%
