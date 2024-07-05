"""
File that has all paths to the different types of analysis from EWA dataset.
"""

import os
import pandas as pd
from constants import *
from functions import *
import math


seed_number = 42 # Seed number defined to keep sample choises the same for all runs. 

# Import metadata files: 
files_info_path = os.path.join(personal_path_to_EWA_DB,'FILES.TSV')
files_info=pd.read_csv(files_info_path,sep='\t')

metadata_path = os.path.join(personal_path_to_EWA_DB,'SPEAKERS.TSV')
metadata=pd.read_csv(metadata_path,sep='\t')

files_info = files_info.rename(columns={"SPEAKER_CODE": "ID"})
metadata = metadata.rename(columns={"SPEAKER_CODE": "ID"})


# Only look at PD and HC
metadata = metadata[(metadata['DIAGNOSIS'] == "Healthy") | (metadata['DIAGNOSIS'] == "Parkinson")]

# Only use those without noise that has publish agreement and is inside the inclusive criteria. 
metadata = metadata[(metadata['LOW_QUALITY'] == False) & (metadata['PUBLISH_AGREEMENT'] == True) & (metadata['INCLUSIVE_CRITERIA'] == True)]

# Updating files info to metada: 
files_info = files_info.loc[files_info['ID'].isin(list(metadata["ID"]))] # Get files_info only about the ID people included in the adapted metadata info. 

"""
### All paths before removing those with to short pitch
"""
paths_EWA_all_with_short_pitch_sequences = list(files_info["AUDIOFILE"])
paths_EWA_all_with_short_pitch_sequences = [elem[1:] for elem in paths_EWA_all_with_short_pitch_sequences]


"""
### All paths after removing people that has utterances that has for short sequence or pitch sequence to calculate features. 

Found with this code after extracting features from paths_EWA_all_with_short_pitch_sequences: 
nan_rows = features[features[phonation_all_features].isna().all(axis=1)]
ids_with_all_nan = nan_rows['ID'].tolist()
unique_ids_with_all_nan = list(set(ids_with_all_nan))

"""
to_short_ids = ['kixjd0bc01', 'kixj1jpf01', 'kixjxipf01', 'o2kgwn7f01', 's9cqfgwi01', 'kixjqw6101', 'kixjv6sk01', 'c8ij11da01', 'o2kgwvif01', 'v0guc40x01', 'ji6qnia901', 'wsnb49v701', 'tbc5sonu01', 'c8ijv96y01', 'wsnbqv0b01', 'wsnb81r301', 'ji6qr0ip01', 'ysw99il501', 'ysw9nq6z01', 'kixjk1cm01', 'kixjlily01', 'tbc55x4e01', 'uc9q88l401', 'ysw9pjx701', 'tbc5l7ki01', 'tbc5l2o301', 'wsnbaiwp01', 'wsnbotws01', '665up78h01', 'frvbabci01', 'kixjv6vb01', 'tbc551k601', 'tbc5bu2q01', 'tbc57wmh01', 'tbc57al401', 'ysw9v4uw01', 's9cqzesh01', 'fgl9e0yp01', 'wsnbcqjk01', 'u1und5jx01', 'b2yuto4l01', 'ysw9316r01', 'kixj4z5c01', 'c8ijgydn01', 'tbc5ht7j01', 'c8ij8z4w01', 'kixjw6to01', 'kixjx4j801', 'wsnb0xw901', 'd1aj3td201', 'kixj1r4y01', 'tbc5psfx01', 'kixjagrh01', 'kixjq95t01']

metadata = metadata[~metadata['ID'].isin(to_short_ids)] 
files_info = files_info.loc[files_info['ID'].isin(list(metadata["ID"]))] 

paths_EWA_all = list(files_info["AUDIOFILE"])
paths_EWA_all = [elem[1:] for elem in paths_EWA_all]


"""
### Removing those that do not have 69 utterances: all_metadata_69_utterances
"""

counts = files_info.groupby('ID').size()  # Group by 'id' and count occurrences
valid_ids = counts[counts == 69].index # Filter the groups where the count is 69

# Filter the DataFrame's based on valid ids
metadata_69_utterances = metadata[metadata['ID'].isin(valid_ids)] 
files_info_69 = files_info.loc[files_info['ID'].isin(list(metadata_69_utterances["ID"]))] 


"""
### Removing those that do not have 100 people: all_metadata_100_people
"""

# Randomly sample 25 females and 25 males from Parkinson patients
parkinson_female = metadata_69_utterances[(metadata_69_utterances['DIAGNOSIS'] == 'Parkinson') & (metadata_69_utterances['SEX'] == 'female')].sample(n=25, random_state=seed_number)
parkinson_male = metadata_69_utterances[(metadata_69_utterances['DIAGNOSIS'] == 'Parkinson') & (metadata_69_utterances['SEX'] == 'male')].sample(n=25, random_state=seed_number)

# Randomly sample 25 females and 25 males from Healthy individuals 
healthy_female = metadata_69_utterances[(metadata_69_utterances['DIAGNOSIS'] == 'Healthy') & (metadata_69_utterances['SEX'] == 'female')].sample(n=25, random_state=seed_number)
healthy_male = metadata_69_utterances[(metadata_69_utterances['DIAGNOSIS'] == 'Healthy') & (metadata_69_utterances['SEX'] == 'male')].sample(n=25, random_state=seed_number)

metadata_100_people = pd.concat([parkinson_female, parkinson_male, healthy_female, healthy_male]) # Concatenate the different samples
files_info_100_people = files_info_69.loc[files_info_69['ID'].isin(list(metadata_100_people["ID"]))] # Get files_info only about the ID people included in the adapted metadata info. 

paths_EWA_100 = list(files_info_100_people["AUDIOFILE"])
paths_EWA_100 = [elem[1:] for elem in paths_EWA_100]

ids_EWA_100 = list(metadata_100_people["ID"])
