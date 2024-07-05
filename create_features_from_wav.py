"""
Script to run to extract features from all wav files in a folder structure. 
Run file with: python EWA_create_features_from_wav.py "Name_of_sheet_for_features_to_be_saved_to".

Creates an excell file (or sheet) and saves all Articulation, Phonation and Prosody features to this sheet. 

Might need to change variables about what folder to extract features from and which paths to use before using the script Can be changed here, or in the constants file. 

This file can also be adapted to work for dynamic features as well. 
"""

import os
import sys
import warnings

from constants import *
from GITA_wav_paths import paths_GITA_all
from EWA_wav_paths import paths_EWA_all, paths_EWA_100, paths_EWA_balanced_69
from functions import *

os.environ['KALDI_ROOT'] = personal_path_to_kaldi
warnings.filterwarnings("ignore", category=RuntimeWarning)  # To remove warning about potentially not getting reliable results. Done to get a cleaner write out. 
                                                            # Warning that creates NaN values (mostly for prosody).
warnings.simplefilter('ignore', np.RankWarning) # Warning we get when finding dynamic prosody features. 
warnings.filterwarnings("ignore", message=".*skipping.*")  # WavFileWarning: Chunk (non-data) not understood, skipping it. Ignoring message about not being able to read all meta data from wav file. (But this should have no influence on the audio signal.) 

sys.path.append("../")                                   

from disvoice.articulation.articulation import Articulation
from disvoice.phonation.phonation import Phonation
from disvoice.prosody.prosody import Prosody

############### VARIABLES TO CHANGE IF WANTED ###############

paths = paths_GITA_all # paths_EWA_100 or paths_GITA_all # Defines the paths we will be getting data from. 
dataset = "EWA" # Eiter GITA or EWA. Due to different layout of the dataset these needs a different method to extract the features. 

new_sheet_name = sys.argv[1] # Defines the name of the sheet that will be created and all the features will be saved in. Takes 1. argument given when running the script from the terminal. 

###################################################

phonationf = Phonation()
articulationf = Articulation()
prosodyf = Prosody()

excell_path = personal_path_to_code + "/Features.xlsx"

features = pd.DataFrame()

if dataset == "EWA": # EWA --> Extracting files seperatly 
    waveform_paths = extend_paths(paths.copy(), personal_path_to_EWA_DB)

    for i in tqdm(range(len(waveform_paths)), total=len(waveform_paths), desc="Extracting features from paths"): # Iterate through paths 
        phonation_features = phonationf.extract_features_file(waveform_paths[i], static=True, plots=False, fmt="csv")
        articulation_features = articulationf.extract_features_file(waveform_paths[i], static=True, plots=False, fmt="csv")
        prosody_features = prosodyf.extract_features_file(waveform_paths[i], static=True, plots=False, fmt="csv")
        
        path_parts = waveform_paths[i].split(personal_path_to_EWA_DB + "/")[1].split('/') # Example: Healthy/611xpfbe01//pataka/pataka.wav
    
        # Adding seperate information to be used to sort out data: 
        information = pd.DataFrame({"ID": [path_parts[1]], "Utterance": [path_parts[4].split('.')[0]], "Utterance type": [path_parts[3]], "Group": [0 if path_parts[0] == "Healthy" else 1]})
        
        # Adding information to features dataframe: 
        feature_one_row = pd.concat([phonation_features, articulation_features, prosody_features, information], axis=1)
        features = pd.concat([features, feature_one_row], axis=0, ignore_index=True) # Combine features
        
        
if dataset == "GITA": # PC-GITA --> Extracting folders together. 
    waveform_paths = extend_paths(paths.copy(), path_PC_GITA_16k)   
    utterance_type = []

    for i in range(len(waveform_paths)):        
        # Extracting features: 
        phonation_features = phonationf.extract_features_path(waveform_paths[i] + "/", static=True, plots=False, fmt="csv")
        articulation_features = articulationf.extract_features_path(waveform_paths[i] + "/", static=True, plots=False, fmt="csv")
        prosody_features = prosodyf.extract_features_path(waveform_paths[i] + "/", static=True, plots=False, fmt="csv")
               
        # Adding information to features dataframe: 
        feature_one_row = pd.concat([phonation_features.drop(['id'], axis=1), articulation_features.drop(['id'], axis=1), prosody_features], axis=1)
        features = pd.concat([features, feature_one_row], axis=0, ignore_index=True) # Combine features 
        
        # Keeping information about utterance type to add later on. 
        utterance_type.extend([waveform_paths[i].split(path_PC_GITA_16k + "/")[1].split("/")[0]] * feature_one_row.shape[0])
        
    # Adding seperate information to be used to sort out data: 
    id_list = list(features['id'])
    utterance = list(features['id'])
    diagnosis = list(features['id'])
    for i in range(len(id_list)):
        utterance[i] = id_list[i].split("_")[1].split(".wav")[0]
        id_list[i] = id_list[i].split("_")[0].replace('AVPEPUDE','')
        diagnosis[i] = [0 if "C" in id_list[i] else 1][0]
        
    features.insert(len(features.columns),'ID',id_list, True)
    features.insert(len(features.columns),'Utterance',utterance, True)
    features.insert(len(features.columns),'Utterance type',utterance_type, True)
    features.insert(len(features.columns),'Group',diagnosis, True)

    features.drop(columns=features.filter(like="id").columns, inplace=True)
    

# Write to excell: 
exel_writer = pd.ExcelWriter(excell_path, mode="w") if not os.path.exists(excell_path) else pd.ExcelWriter(excell_path, mode="a", if_sheet_exists = "new")

with exel_writer as writer:
    features.to_excel(writer, sheet_name=new_sheet_name, header=True)
    
###################################################