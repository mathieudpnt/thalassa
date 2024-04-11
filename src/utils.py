import os
import re
import numpy as np
import pandas as pd
import glob
from scipy import signal
import cv2 as cv


def click_train_extraction(data, Fe, f_min, f_max, ratio, energy_threshold, ICI_max, nb_click_min, height_morph, width_morph, width_min):
    '''L'objectif de cette fonction, est de partir d'un signal de départ et de retourner une liste des intervalles qui
    contiennent les potentiels trains de clics de ce signal. Dans notre cas, ce signal d'entrée est le résultat de
    l'intercorrélation du signal de départ "sig" par l'une des ondelettes-mères. On note que la liste retournée contient
    des intervalles matriciels, qui seront ensuite traduits en intervalles temporels par la fonction « extraction »

    Parameters
    -------

    data: float64 array
        the product of the cross correlation between the signal and the chosen model

    Fe: int
        Sampling frequency

    f_min: int;float
        Min frequency of study on the spectrogram

    f_max: int;float
        Max frequency of study on the spectrogram

    ratio: int
        Value used for the homothety

    height_morph: int
        Height of the structuring element to search for click pattern

    width_morph: int
        Width of the structuring element to search for click pattern

    energy_threshold: int
        % of energy values that are ignored

    ICI_max: float
        Maximum ICI interval between two consecutive click from the same click train

    width_min: int
        Minimal click width

    nb_click_min: int
        Smallest number of click to define a click train

    Returns
    -------

    click_train: List
        List of time intervals of the detected click trains
    '''
    N = len(data)
    duration = N / Fe

    # Spectrogram
    overlap = 0.95
    nperseg = 2048
    f, t, Sxx = signal.spectrogram(data, Fe, nperseg=nperseg, window='hamming', noverlap=round(overlap * nperseg))

    # length of the spectrogram as an image
    len_sig = len(t)
    # height of the spectrogram as an image
    height_sig = len(f)

    # equilalent f_min and f_max in the Sxx matrix
    y_min = round(f_min * height_sig / f_max)
    y_max = height_sig

    # Spectrogram between f_min and f_max
    mat = Sxx[y_min:y_max, :]

    # Homothety, this will range the matrix values from 0 to ratio
    m = np.min(mat)  # smallest value of spectrogram
    M = np.max(mat)  # highest value of spectrogram
    mat_homothety = np.round(mat * ratio / (M - m) + ratio * m / (M - m))

    # Binary matrix, only the values of mat_homothety that are above the energy threshold are kept
    countscum = np.cumsum(np.histogram(mat_homothety, bins=(ratio + 1))[0]) / (len_sig * (y_max - y_min))  # cumulative histogram normalized by the total number of elements
    limit = np.argmax(countscum > energy_threshold / 100)  # Find the index where the cumulative energy surpasses the threshold
    mat_binary = mat_homothety > limit  # Threshold the binary matrix based on the limit

    # structure for click extraction
    # click_structure = np.ones((1, 215), dtype=np.uint8)
    click_structure = cv.getStructuringElement(cv.MORPH_RECT, (width_morph, height_morph))
    mat_morph = cv.morphologyEx(np.float32(mat_binary), cv.MORPH_OPEN, click_structure)

    # To retrieve time intervals where a click pattern is located, the columns of mat_morph != 0 are stored
    mat_sum = np.sum(mat_morph, axis=0)

    # Click train extraction

    # This loop search for click patterns
    # A click pattern is defined as a serieof at least
    # non-null consecutive values of length >= width_min
    click_candidate, click = [], []
    for i in range(len(mat_sum)):
        if mat_sum[i] != 0:
            click_candidate.append(i)
        else:
            if len(click_candidate) > width_min:
                click.append([click_candidate[0], click_candidate[-1]])
                click_candidate = []  # Reset train start

    # This second loop search now for click train with the clicks found in the previous loop
    # A click train is defined as a serie of at least nb_click_min consecutive clicks
    # with an ICI <= ICI_max. As the data is computed in a matrix form (not temporal values),
    # ICI_max is translated into its corresponding matrix counterpart : delta
    click_train_candidate, click_train = [], []
    delta = round(ICI_max * len(mat_sum) / duration)
    nb_click = 1

    for i in range(len(click) - 1):

        # Interval between the end of click[i] and the start of click[i+1]
        interval = click[i + 1][0] - click[i][1]

        # The ICI is smaller or equal than ICI_max
        if interval <= delta:

            if not click_train_candidate:
                click_train_candidate = [click[i][0], click[i + 1][1]]
            else:
                click_train_candidate = [min(click_train_candidate[0], click[i][0]), max(click_train_candidate[1], click[i + 1][1])]

            nb_click += 1
            # print(f'i={i}, candidate:{click_train_candidate}, {nb_click} clicks')

        # The ICI is greater than ICI_max
        else:
            # The minimal click number requirement is met: it is a click train
            if nb_click >= nb_click_min:
                click_train.append(click_train_candidate)
                # print(f'i={i}, candidate:{click_train_candidate}, {nb_click} clicks -> click train registred')

            # The minimal click number requirement is not met: it is not a click train
            else:
                # print(f'i={i}, candidate:{click_train_candidate}, {nb_click} clicks -> click train not registred')
                pass

            # Reset the indicators
            click_train_candidate = []
            click_train_candidate = [click[i + 1][0], click[i + 1][1]]
            nb_click = 1
            # print(f'i={i}, new candidate:{click_train_candidate}, {nb_click} click')

        # Check for a train click on the last iteration of the loop
        if i == len(click) - 2 and nb_click >= nb_click_min:
            click_train.append(click_train_candidate)
            # print(f'i={i}, candidate:{click_train_candidate}, {nb_click} clicks -> click train registred')

    # From matrix value to corresponding temporal values
    detection = []
    for train in click_train:
        detection.append([round((train[0] * N / len_sig) / Fe, 3), round((train[1] * N / len_sig) / Fe, 3)])

    return detection


def gabor(model, N, dur):
    '''Selection of the gabor wave model to pcompute the crossed-correlation

    Parameters
    -------
        model: int
            chosen model

        N: int
            number of samples

        dur: float
            duration of the signal
    '''

    match model:
        case 1:
            T = np.linspace(0, dur, N)
            f_gabor = 40000  # Fréquence centrale de l'onde de Gabor ex : 40 kHz
            sigma_gabor = 0.00001  # Écart-type de l'onde de Gabor ex : 10 microsecondes
            return np.exp(- (T - dur / 2) ** 2 / (2 * sigma_gabor ** 2)) * np.cos(2 * np.pi * f_gabor * (T - dur / 2))
        case 2:
            f_gabor_bis = 40000  # Fréquence centrale de l'onde de Gabor ex : 40 kHz
            sigma_gabor_bis = 2e-5  # Écart-type de l'onde de Gabor ex : 20 microsecondes
            T = np.linspace(0, dur, N)
            fct_1 = np.cos(2 * np.pi * f_gabor_bis * (T - (5))) / 15
            fct_2 = np.exp(- (T - (dur / 2) ** 2 / (2 * sigma_gabor_bis ** 2))) * 10 / 15
            fct_3 = np.concatenate((np.ones(int(N / 2)), np.exp(-74000 * (T - (dur / 2))[int(N / 2)::])), axis=None)
            return fct_1 * fct_2 * fct_3 * 150
        case 3:
            moy_periode = 2.66e-5   # période moyene au sein d'un seul clic, entre les différents maxima
            frequence_onde = 1 / moy_periode    # fréquence associée à "moy_periode"

            F_mini = 2 * frequence_onde  # on comprend qu'il faut au minimum 2 données par période pour la création de cette onde
            # en effet, il faut : 1 point pour le maximum et un point pour le minimum pour chaque période, on aura donc bien une onde
            # d'aspect triangulaire
            Fe = round(N / dur)

            num = round(Fe / F_mini)  # rapport entier entre "Fe" et "F_mini" qui détermine le nombre de points par demi-période lors
            # de la création de l'onde "onde_gabor_ter"

            # L'expérience montre que le second pic d'un clic possède toujurs la plus grande amplitude, on s'en sert donc comme référence
            # pour définir les hauteurs relatives des autres pics (pic 1, pic 3 et pic 4) :

            rel1 = 0.28585  # moyenne de la hauteur relative entre l'amplitude du premier pic et celle du second pic
            rel3 = 0.62505  # moyenne de la hauteur relative entre l'amplitude du troisième pic et celle du second pic
            rel4 = 0.27160  # moyenne de la hauteur relative entre l'amplitude du quatrième pic et celle du second pic

            onde = [0, 0, rel1 / 2]
            h_relatives = [rel1, 1, rel3, rel4]

            if num == 1:
                onde = [0, 0, rel1 / 2, -rel1 / 2, 0.5, -0.5, rel3 / 2, -rel3 / 2, rel4 / 2, -rel4 / 2]
            else:
                for p in range(4):
                    for k in range(num):
                        onde.append(onde[-1] - h_relatives[p] / 2)
                    if p != 3:
                        onde.append(h_relatives[p + 1] / 2)
            onde.append(0)
            onde.append(0)
            return np.concatenate((np.zeros(int((N - len(onde)) / 2)), onde, np.zeros(int((N - len(onde)) / 2))), axis=None)
        case _:
            raise ValueError(f'Model not well defined: {model}')


def extract_datetime(var: str, formats=None) -> pd.Timestamp:
    """Extracts datetime from filename based on the date format

    Parameters
    -------
    var: 'str'
            name of the wav file
        formats: 'str'
            The date template in strftime format.
            For example, `2017/02/24` has the template `%Y/%m/%d`
            For more information on strftime template, see https://strftime.org/
    Returns
    -------
        date_obj: pd.Timestamp
            datetime corresponding to the datetime found in var
    """

    if formats is None:
        # add more format if necessary
        formats = [
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",
            r"\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}",
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
            r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}",
            r"\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}",
            r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}",
        ]
    match = None
    for f in formats:
        match = re.search(f, var)
        if match:
            break
    if match:
        dt_string = match.group()
        if f == r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}":
            dt_format = "%Y-%m-%dT%H-%M-%S"
        elif f == r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}":
            dt_format = "%Y-%m-%d_%H-%M-%S"
        elif f == r"\d{2}\d{2}\d{2}\d{2}\d{2}\d{2}":
            dt_format = "%y%m%d%H%M%S"
        elif f == r"\d{2}\d{2}\d{2}_\d{2}\d{2}\d{2}":
            dt_format = "%y%m%d_%H%M%S"
        elif f == r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}":
            dt_format = "%Y-%m-%d %H:%M:%S"
        elif f == r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}":
            dt_format = "%Y-%m-%dT%H:%M:%S"
        elif f == r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}":
            dt_format = "%Y_%m_%d_%H_%M_%S"
        elif f == r"\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2}":
            dt_format = "%Y_%m_%dT%H_%M_%S"
        elif f == r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}":
            dt_format = "%Y%m%dT%H%M%S"

        date_obj = pd.to_datetime(dt_string, format=dt_format)
        return date_obj
    else:
        raise ValueError(f"{var}: No datetime found")
