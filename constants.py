
import os


# Constants to change according to your setup: 
personal_path_to_disvoice = '/home/'+os.environ['USER']+'/master-thesis/DisVoice'

personal_path_to_kaldi = '/home/'+os.environ['USER']+'/master-thesis/kaldi' 

personal_path_to_PC_GITA = '/home/'+os.environ['USER']+'/master-thesis/PC-GITA-v2-mod' 

personal_path_to_EWA_DB = '/home/'+os.environ['USER']+'/master-thesis/S0489' 

personal_path_to_code = '/home/'+os.environ['USER']+'/master-thesis/Master-thesis-about-Parkinsons-Detection'

personal_path_to_balanced_folds = personal_path_to_code + "/Balanced_folds"
personal_path_to_results = personal_path_to_code + "/Generated_results"

path_PC_GITA_16k = personal_path_to_PC_GITA + "/PC-GITA_per_task_16000Hz"

# Color definitions used for plotting
HC_color = 'darkgreen'
PD_color = 'firebrick'

correct_color = 'lightgreen'
wrong_color = 'lightcoral'

female_color = "deeppink"
male_color = "dodgerblue"

# Label mapping between Parkinson's Disease and Healthy Control
class_labels = {"HC": 0, "PD": 1} 
