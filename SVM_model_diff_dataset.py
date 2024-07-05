import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import time

from functions import *
from constants import *
from definition_of_data_and_feature_groups import *

"""
Description of this file: 
Same as SVM_model, but lets us train on a combination of EWA and GITA and test on this combination. 

"""

###########   DEFINING PARAMETERS FOR THE RUN   #################

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
    "Random 100": random_100,
    "SFS 100": sfs_100_features,
}

# The data subset group for GITA and EWA needs to be the same length and every element will be used together for train and test. 
data_subset_GITA = {
    "all data": [GITA_all_data, [""]],
    "continuous speech":  [GITA_cont_speech, [""]],
    "Words": [["Words"], [""]], 
    "Pataka": [["DDK_analysis"], ["pataka"]],
    "A": [["Vowels"], ["a1"]],
    
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

# SVM parameters --> CHosen based on grid search from grid_search
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

metadata_columns =  ["SEX", "AGE"] 
fold_info_columns = ["Fold"]
feature_info_columns = ["ID", "Utterance", "Utterance type", "Group"]

# patient_info_columns will be used later on to remove all metadata and then also want to remove id. 
patient_info_columns = feature_info_columns + metadata_columns + fold_info_columns

# For GITA:
features_GITA = pd.read_excel(feature_path, sheet_name="GITA-all", index_col=0)

metadata_path_GITA = os.path.join(personal_path_to_PC_GITA,'PCGITA_metadata.xlsx')
metadata_GITA = pd.read_excel(metadata_path_GITA)
metadata_GITA = restructure_id(metadata_GITA)
fold_info_path_GITA = os.path.join(personal_path_to_balanced_folds,"kfold-groups-tsv.csv" )  # Path to fold distribution information. 
fold_info_GITA = pd.read_csv(fold_info_path_GITA)

features_GITA = add_columns_to_dataframe(features_GITA, metadata_GITA, metadata_columns) # Only bring along SEX and AGE from metadata. 
features_GITA = add_columns_to_dataframe(features_GITA, fold_info_GITA, fold_info_columns) # Only bring along Fold numhber from fold_info. 
features_GITA = remove_NaN(features_GITA, patient_info_columns, print_out = False) # Changes NaN values to be zero values instead. 
    
# FOR EWA: 
features_EWA = pd.read_excel(feature_path, sheet_name="EWA-100", index_col=0)

metadata_path_EWA = os.path.join(personal_path_to_EWA_DB,'SPEAKERS.TSV')
metadata_EWA=pd.read_csv(metadata_path_EWA,sep='\t') # Get all metadata 

metadata_EWA = metadata_EWA.rename(columns={"SPEAKER_CODE": "ID"}) # Restructure IDs to be called "ID". 
metadata_EWA = metadata_EWA.loc[metadata_EWA['ID'].isin(list(features_EWA["ID"]))] # Extract the data with the IDs from the feature data. 

fold_info_path_EWA = os.path.join(personal_path_to_balanced_folds,"kfold-groups-ewa.csv" )  # Path to fold distribution information. 
fold_info_EWA = pd.read_csv(fold_info_path_EWA)

features_EWA = add_columns_to_dataframe(features_EWA, metadata_EWA, metadata_columns) # Only bring along SEX and AGE from metadata. 
features_EWA = add_columns_to_dataframe(features_EWA, fold_info_EWA, fold_info_columns) # Only bring along Fold numhber from fold_info. 
features_EWA = remove_NaN(features_EWA, patient_info_columns, print_out = False) # Changes NaN values to be zero values instead. 


train_dataset = "GITA"
test_dataset = "EWA"

if train_dataset == test_dataset:
    print("Rather use svm_model file for cases where the training and testing data is from the same dataset. ")
elif train_dataset == "GITA" and test_dataset == "EWA":
    train_features = features_GITA
    train_data_subset = data_subset_GITA
    
    test_features = features_EWA
    test_data_subset = data_subset_EWA
elif train_dataset == "EWA" and test_dataset == "GITA":
    train_features = features_EWA
    train_data_subset = data_subset_EWA
    
    test_features = features_GITA
    test_data_subset = data_subset_GITA
else:
    print("Something went wrong with the defining of training and testing datasets. Try another definition. ")
    

np.random.seed(seed_number) 
run_number = 0

if len(train_data_subset) != len(test_data_subset):
    print("The amount of data training subsets is not the same as the amount of the data test subsets. These needs to be the same, and every comparing will be done with every pair of set.")
else:
    num_data_subsets = len(train_data_subset)
    total_runs = len(feature_subsets)*num_data_subsets

if total_runs <= 0:
    raise ValueError("Not enough data or features are provided. Script stopped. Redefine feature and data subsets.")

results = pd.DataFrame(index=[i for i in range(1, total_runs + 1)], columns=["Feature", "numF", "Data subset", "Train num samples", "Test num samples", "avg TrainAcc", "avg Acc", "avg Sens", "avg Spec"])

if num_data_subsets>1:
    res_accuracies = pd.DataFrame(index=["num samples training", "num samples test"] + list(feature_subsets.keys()), columns=["numF"] + list(train_data_subset.keys()))
else: 
    res_accuracies = pd.DataFrame()

    
for (train_data_name, train_data_values), (test_data_name, test_data_values) in zip(train_data_subset.items(), test_data_subset.items()):  # Run for every data type 
    
    # Extracting only the data for this run: 
    
    # Train data: 
    train_data_for_run = train_features.copy()
    if train_data_values[0]  != [""]: # If defined utterance sub groups (like vowels or words or combinations)
        train_data_for_run = train_data_for_run[train_data_for_run["Utterance type"].isin(train_data_values[0] )]
    if train_data_values[1] != [""]: # If defined specific utterances (like "a" or "viaje" or combinations)
        train_data_for_run = train_data_for_run[train_data_for_run["Utterance"].isin(train_data_values[1])]
        
    # Test data: 
    test_data_for_run = test_features.copy()
    if test_data_values[0] != [""]: # If defined utterance sub groups (like vowels or words or combinations)
        test_data_for_run = test_data_for_run[test_data_for_run["Utterance type"].isin(test_data_values[0])]
    if test_data_values[1] != [""]: # If defined specific utterances (like "a" or "viaje" or combinations)
        test_data_for_run = test_data_for_run[test_data_for_run["Utterance"].isin(test_data_values[1])]
    
    for feature_name, feature_values in feature_subsets.items():  #  Run for every feature type  
        
        start_time = time.time()
        
        # Extracting only the features for this run: 
        train_features_for_run = train_data_for_run.copy()
        train_features_for_run = train_features_for_run.loc[:, feature_values + patient_info_columns]
        train_num_samples = train_features_for_run.shape[0]
        
        test_features_for_run = test_data_for_run.copy()
        test_features_for_run = test_features_for_run.loc[:, feature_values + patient_info_columns]
        test_num_samples = test_features_for_run.shape[0]
        
        num_features = len(feature_values)
               
        # Print out information about run: 
        print("---------------------------------------------------------------")
        run_number += 1
        print(f"Experiment number: {run_number} of {total_runs}")
        print("Features: ", feature_name) 
        print("Data for training: ", train_data_name)
        print("Data for testing: ", test_data_name)
        
        svm_model = SVC(kernel=kernel_best, C=C_best, gamma=gamma_best) # Only use gamma for the rbf kernel 
        
        features_combined = pd.concat([train_features_for_run, test_features_for_run])
        accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores, confusion_mat_sum, metadata_with_classification_result = balanced_cross_validation(data=features_combined, model=svm_model, metadata_columns=patient_info_columns, write_out_each_fold=False)        

        roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set = calculate_classification_detailes_from_cross_fold(metadata_with_classification_result)
        avgAcc_value, _, _, avgAcc, avgTrainAcc, avgSens, avgSpec = calculate_validation_measures_from_cross_fold(accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores)
                    
        
        # Add data to table: 
        results.loc[run_number, ["Feature", "numF", "Data subset", "Train num samples", "Test num samples", "avg Acc value", "avg TrainAcc", "avg Acc", "avg Sens", "avg Spec", "List of trainAcc", "List of Acc", "List of Sens", "List of Spec", "Roc Auc", "Distance to boarder for correct classified", "Distance to boarder for Wrongly classified", "TN", "TP", "FN", "FP", "Best threshold for test set"]] = [feature_name, num_features, train_data_name, train_num_samples, test_num_samples, avgAcc_value, avgTrainAcc, avgAcc, avgSens, avgSpec, str([round(elem * 100, 1) for elem in training_accuracy_scores]), str([round(elem * 100, 1) for elem in accuracy_scores]), str([round(elem * 100, 1) for elem in sensitivity_scores]), str([round(elem * 100, 1) for elem in specificity_scores]), roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set]
        
        if num_data_subsets>1: # If multiple, also create a table with only the accuracy.     
            res_accuracies.at[feature_name, 'numF'] = num_features
            res_accuracies.at['num samples training', train_data_name] = train_num_samples
            res_accuracies.at['num samples test', train_data_name] = test_num_samples # Training and testing data name should be the same so use train_data_name here as well. 
            res_accuracies.at[feature_name, train_data_name] = avgAcc_value
            
        results.at[run_number, 'Num samples combiend'] = train_num_samples + test_num_samples
        res_accuracies.at['Num samples combiend', train_data_name] = train_num_samples + test_num_samples
            
        # Printing out the results:         
        if print_out_each_run: # Print out some basic information for each run        
            print(f"Total time used for experiment numer {run_number}: {round(time.time() - start_time,2)} seconds")    
            print(f"Len of (test_data, train_data, features): ({train_num_samples}, {test_num_samples}, {num_features})")   
            print("Data over cross validation: ")
            write_out_cross_fold_results(number_of_cross_val_folds, accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores, confusion_mat_sum)
            
    
# If we want the data to be sorted by features instead of data. 

if sort_by_feature:
    results = results.sort_values(by=['Feature']) # Default is quicksort. If wanted stable sorting could use mergesort. 

# Print out table based on number of data types.    
print("---------------------------------------------------------------")
print("Table summary for all the runs: ")

printable_results = results[["Feature", "numF", "Data subset", "Train num samples", "Test num samples", "avg Acc value", "avg TrainAcc", "avg Acc", "avg Sens", "avg Spec"]]

if num_data_subsets>1 and print_detailed_summary:
    print(res_accuracies)
    print("---------------------------------------------------------------")
    print("Detailed summary for all the runs: ")
    print(printable_results)    
elif num_data_subsets>1:
    print(res_accuracies)
elif num_data_subsets==1:
    
    print(printable_results)
else: 
    raise ValueError("No data subset was defined. Script stopped, define data subsets and try again.")
        

# Print out the feature - data combination (based on avgAcc) that gives the best result. 
print("---------------------------------------------------------------")
max_row = results.loc[results['avg Acc value'].idxmax()]
print(f"The best feature - data combination (based on avg Acc) among the ones tested is: {max_row['Feature']} and {max_row['Data subset']} with an an avg Acc on {max_row['avg Acc']}")

acc_sorted_results = results.sort_values(by=['avg Acc value'], ascending=False)

# Save data to excell 
if save_to_excell:
    best_feature_data = pd.DataFrame({'Best feature': [max_row['Feature']], 'Best Data': [max_row['Data subset']], 'avg Acc': [max_row['avg Acc']]})
    parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'seed_number', 'kernel_best', 'C_best', 'gamma_best', 'number_of_cross_val_folds', 'print_out_each_run',  'print_detailed_summary'], 'Value': ["svm model diff dataset", seed_number, kernel_best, C_best, gamma_best, number_of_cross_val_folds, print_out_each_run, print_detailed_summary]})
    features_to_exel = pd.DataFrame(feature_subsets.items(), columns=['Subset', 'Features'])
    train_data_to_excel = pd.DataFrame(train_data_subset.items(), columns=['Train subset', 'Train data'])
    test_data_to_excel = pd.DataFrame(test_data_subset.items(), columns=['Test subset', 'Test data'])
    
    write_to_excel([res_accuracies, results, best_feature_data, acc_sorted_results, parameters_to_exel, features_to_exel, train_data_to_excel, test_data_to_excel], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell

print("The data showed here is for EWA and GITA data combined (training and testing data combined) and used in cross validation together as one dataset. ")
#####################################################################

