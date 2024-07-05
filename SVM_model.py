import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import time

from functions import *
from constants import *
from definition_of_data_and_feature_groups import *

"""
- The file will run an experiment with cross validation and give out the results for all feature and data combinations that are defined. 


More detailed description of this file: 
- Define feature and data sub sets to do runs with. All combinations of these will then be run. 
- Define model parameters and seed_number. 
- Traines a model on the different feature - data combinations and predicts results using 10 fold cross validation. 
- Can choose to print out more for each fold if print_out_each_run variable is True. If False we only print out the table at the end. 
    - Note: If you want images to be shown - run in interactive window. Only needed if print_out_each_run is True.    
    
- Use cases: Can either have 1 feature and 1 datatype or multiple of either features or datatypes, or multiple of both.  Table will be printed out with only accuracy if mulitple data types are defined,
    but with avg TrainAcc, avg Acc, avg Sens, avg Spec as well if only one datatype is defined or if print_detailed_summary is True. 

"""


###########   DEFINING PARAMETERS FOR THE RUN   #################

feature_sheet_name = "GITA-all" # "GITA-all" # "EWA-100" # EWA-balanced-69 # Sheet name we want to get features from. 
dataset = "GITA" # Eiter GITA or EWA.
fold_info_file = "kfold-groups-tsv.csv" # ewa, tsv or mf

# Define dictionaries for feature and data groups and their corresponding name as a key // Note: Need to define at least one for each of them.
    
feature_subsets = {
    "all features": all_features,
    "articulation": articulation_all_features,
    "phonation": phonation_all_features,
    "prosody": prosody_all_features,
    "ZOF - Zero optimized": ZOF,
    "FBF - Frequency Based": FBF,
    "SOAF - State of the art features": SOTAF_all,
    "SOAF_avg - State of the art avg features": SOTAF_avg,
    # "RFE_100": rfe_100_features,
    "Random 100": random_100,
    "SFS 100": sfs_100_features,
}

# Possible utterance types:  "DDK_analysis", "modulated_vowels", "monologue", "read_text", "sentences", "sentences2", "Vowels", "Words"
data_subset_GITA = {
    "all data": [GITA_all_data, [""]],
    "continuous speech":  [GITA_cont_speech, [""]],
    "all vowels": [["Vowels", "modulated_vowels"], [""]],
    "sustained vowels": [["Vowels"], [""]],
    "Words": [["Words"], [""]], 
    "DDK": [["DDK_analysis"], [""]],
    "A": [["Vowels"], ["a1", "a2", "a3"]],
    "U": [["Vowels"], ["u1", "u2", "u3"]],
    # "pataka": [["DDK_analysis"], ["pataka"]],
    # "A1": [["Vowels"], ["a1"]],
} # In the form: "Name": [utterance type, specific utterance] --> Example: ["a1", "viaje"] --> Look at possible data and features.


data_subset_EWA = {
    "all data": [EWA_all_data, [""]],
    "continuous speech":  [["picture"], [""]],
    "Words": [["naming"], [""]],
    "Pataka": [["pataka"], [""]],
    "A": [["phonation"], [""]],
}

# Seed number 
seed_number = 42

# SVM parameters --> Choosen based on grid search from grid_search
kernel_best ='rbf' 
C_best = 100
gamma_best = 0.0001

number_of_cross_val_folds = 10  # Need to change the fold info if something else than 10 is wanted. 

# Print and file saveing settings: 
print_out_each_run = False # If true, we print out some more detailed information for each run. 
print_detailed_summary = True # If true then it print out a table with more than only accuracy for cases where we are testing for different data sub sets. 
save_to_excell = True  # If true it writes to Automated_results excell file

sort_by_feature = False # Detailed result dataframe will be sorted by features if True, and by Data if False. 

#####################################################################




###########   CODE   ################################################

feature_path = personal_path_to_code + '/Features.xlsx'   # Defines path to feature folder to use. 
features = pd.read_excel(feature_path, sheet_name=feature_sheet_name, index_col=0)

if dataset == "GITA":
    data_subset = data_subset_GITA
    metadata_path = os.path.join(personal_path_to_PC_GITA,'PCGITA_metadata.xlsx')
    metadata = pd.read_excel(metadata_path)
    metadata = restructure_id(metadata)
    
elif dataset == "EWA": 
    data_subset = data_subset_EWA
    
    # Import metadata files: 
    metadata_path = os.path.join(personal_path_to_EWA_DB,'SPEAKERS.TSV')
    metadata=pd.read_csv(metadata_path,sep='\t') # Get all metadata 
    
    metadata = metadata.rename(columns={"SPEAKER_CODE": "ID"}) # Restructure IDs to be called "ID". 
    metadata = metadata.loc[metadata['ID'].isin(list(features["ID"]))] # Extract the data with the IDs from the feature data. 

                   
else: 
    print("No aproved dataset is defined. Choose either GITA or EWA")

fold_info_path = os.path.join(personal_path_to_balanced_folds,fold_info_file)  # Path to fold distribution information. 
fold_info = pd.read_csv(fold_info_path)

metadata_columns =  ["SEX", "AGE"] # The rest of the elements will rather be looked into in detailed model file. # ["SEX", "AGE", "UPDRS", "UPDRS-speech", "H/Y", "time after diagnosis"] # Use this as definition on what to remove later on. 
fold_info_columns = ["Fold"]
feature_info_columns = ["ID", "Utterance", "Utterance type", "Group"]

# patient_info_columns will be used later on to remove all metadata and then also want to remove id. 
patient_info_columns = feature_info_columns + metadata_columns + fold_info_columns

# Add metadata and fold info: 
features = add_columns_to_dataframe(features, metadata, metadata_columns) # Only bring along SEX and AGE from metadata. 
features = add_columns_to_dataframe(features, fold_info, fold_info_columns) # Only bring along Fold numhber from fold_info. 

features = remove_NaN(features, patient_info_columns, print_out = False) # Changes NaN values to be zero values instead. 

np.random.seed(seed_number) 

run_number = 0
total_runs = len(feature_subsets)*len(data_subset)

if total_runs <= 0:
    raise ValueError("Not enough data or features are provided. Script stopped. Redefine feature and data subsets.")

results = pd.DataFrame(index=[i for i in range(1, total_runs + 1)], columns=["Feature", "numF", "Data", "num samples", "avg TrainAcc", "avg Acc", "avg Sens", "avg Spec"])

if len(data_subset)>1:
    res_accuracies = pd.DataFrame(index=["num samples"] + list(feature_subsets.keys()), columns=["numF"] + list(data_subset.keys()))
else: 
    res_accuracies = pd.DataFrame()

for data_name, data_values in data_subset.items(): # Run for every data type 
    
    # Extracting only the data for this run: 
    data_for_run = features.copy()
    
    utterance_type = data_values[0] 
    if utterance_type != [""]: # If defined utterance sub groups (like vowels or words or combinations)
        data_for_run = data_for_run[data_for_run["Utterance type"].isin(utterance_type)]

    specific_utterance = data_values[1]
    if specific_utterance != [""]: # If defined specific utterances (like "a" or "viaje" or combinations)
        data_for_run = data_for_run[data_for_run["Utterance"].isin(specific_utterance)]
    
    for feature_name, feature_values in feature_subsets.items():  #  Run for every feature type  
        
        start_time = time.time()
        
        # Extracting only the features for this run: 
        features_for_run = data_for_run.copy()
        features_for_run = features_for_run.loc[:, feature_values + patient_info_columns]
                
        num_samples = features_for_run.shape[0]
        num_features = len(feature_values)
               
        # Print out information about run: 
        print("---------------------------------------------------------------")
        run_number += 1
        print(f"Experiment number: {run_number} of {total_runs}")
        print("Features: ", feature_name) 
        print("Data: ", data_name)
        
        svm_model = SVC(kernel=kernel_best, C=C_best, gamma=gamma_best) # Only use gamma for the rbf kernel 
        
        accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores, confusion_mat_sum, metadata_with_classification_result = balanced_cross_validation(data=features_for_run, model=svm_model, metadata_columns=patient_info_columns, write_out_each_fold=False)        
    
        roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set = calculate_classification_detailes_from_cross_fold(metadata_with_classification_result)
    
        avgAcc_value, _, _, avgAcc, avgTrainAcc, avgSens, avgSpec = calculate_validation_measures_from_cross_fold(accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores)
        
        # Add data to table: 
        results.loc[run_number, ["Feature", "numF", "Data", "num samples", "avg Acc value", "avg TrainAcc", "avg Acc", "avg Sens", "avg Spec", "List of trainAcc", "List of Acc", "List of Sens", "List of Spec", "Roc Auc", "Distance to boarder for correct classified", "Distance to boarder for Wrongly classified", "TN", "TP", "FN", "FP", "Best threshold for test set"]] = [feature_name, num_features, data_name, num_samples, avgAcc_value, avgTrainAcc, avgAcc, avgSens, avgSpec, str([round(elem * 100, 1) for elem in training_accuracy_scores]), str([round(elem * 100, 1) for elem in accuracy_scores]), str([round(elem * 100, 1) for elem in sensitivity_scores]), str([round(elem * 100, 1) for elem in specificity_scores]), roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set]
                     
        if len(data_subset)>1: # If multiple, also create a tablel with only the accuracy.     
            res_accuracies.at[feature_name, 'numF'] = num_features
            res_accuracies.at['num samples', data_name] = num_samples
            res_accuracies.at[feature_name, data_name] = avgAcc_value
                
        # Printing out the results:         
        if print_out_each_run: # Print out some basic information for each run        
            print(f"Total time used for experiment numer {run_number}: {round(time.time() - start_time,2)} seconds")    
            print(f"Len of (data, features): ({num_samples}, {num_features})")   
            print("Data over cross validation: ")
            write_out_cross_fold_results(number_of_cross_val_folds, accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores, confusion_mat_sum)
            
    
# If we want the data to be sorted by features instead of data. 
if sort_by_feature:
    results = results.sort_values(by=['Feature']) # Default is quicksort. If wanted stable sorting could use mergesort. 


# Print out table based on number of data types.    
print("---------------------------------------------------------------")
print("Table summary for all the runs: ")

printable_results = results[["Feature", "numF", "Data", "num samples", "avg Acc value", "avg TrainAcc", "avg Acc", "avg Sens", "avg Spec"]]

if len(data_subset)>1 and print_detailed_summary:
    print(res_accuracies)
    print("---------------------------------------------------------------")
    print("Detailed summary for all the runs: ")
    print(printable_results)    
elif len(data_subset)>1:
    print(res_accuracies)
elif len(data_subset)==1:
    
    print(printable_results)
else: 
    raise ValueError("No data subset was defined. Script stopped, define data subsets and try again.")
        

# Print out the feature - data combination (based on avgAcc) that gives the best result. 
print("---------------------------------------------------------------")
max_row = results.loc[results['avg Acc value'].idxmax()]
print(f"The best feature - data combination (based on avg Acc) among the ones tested is: {max_row['Feature']} and {max_row['Data']} with an an avg Acc on {max_row['avg Acc']}")

acc_sorted_results = results.sort_values(by=['avg Acc value'], ascending=False)

# Save data to excell 
if save_to_excell:
    best_feature_data = pd.DataFrame({'Best feature': [max_row['Feature']], 'Best Data': [max_row['Data']], 'avg Acc': [max_row['avg Acc']]})
    parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'Dataset', 'seed_number', 'kernel_best', 'C_best', 'gamma_best', 'number_of_cross_val_folds', 'print_out_each_run',  'print_detailed_summary', 'Feature sheet name', 'Fold info file name'], 'Value': ["svm_model", dataset, seed_number, kernel_best, C_best, gamma_best, number_of_cross_val_folds, print_out_each_run, print_detailed_summary, feature_sheet_name, fold_info_file]})
    features_to_exel = pd.DataFrame(feature_subsets.items(), columns=['Subset', 'Features'])
    data_to_exel = pd.DataFrame(data_subset.items(), columns=['Subset', 'Data'])
    
    write_to_excel([res_accuracies, results, best_feature_data, acc_sorted_results, parameters_to_exel, features_to_exel, data_to_exel], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell

#####################################################################

