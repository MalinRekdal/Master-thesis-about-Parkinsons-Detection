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
Same as SVM_model, but uses all data from two datasets (GITA and EWA) for training set and testing. Instead of using cross validation.
"""


###########   DEFINING PARAMETERS FOR THE RUN   #################

train_dataset = "EWA"
test_dataset = "GITA"
def_experiment = "Using all data for training and testing for each of the sub sets here. No cross validation"

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

metadata_columns =  ["SEX", "AGE"] # The rest of the elements will rather be looked into in detailed model file. # ["SEX", "AGE", "UPDRS", "UPDRS-speech", "H/Y", "time after diagnosis"] # Use this as definition on what to remove later on. 
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


if train_dataset == test_dataset:
    print("Rather use svm_model file for cases where the training and testing data is from the same dataset. ")
elif train_dataset == "GITA" and test_dataset == "EWA":
    # Train things - GITA 
    train_features = features_GITA
    train_data_subset = data_subset_GITA
    
    # Test things - EWA 
    test_features = features_EWA
    test_data_subset = data_subset_EWA
elif train_dataset == "EWA" and test_dataset == "GITA":
    # Train things - EWA 
    train_features = features_EWA
    train_data_subset = data_subset_EWA
    
    # Test things - GITA 
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

results = pd.DataFrame(index=[i for i in range(1, total_runs + 1)], columns=["Feature", "numF", "Data subset", "Train num samples", "Test num samples", "TrainAcc", "Acc", "Sens", "Spec"])

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
        
        print(f"Total time used for experiment numer {run_number}: {round(time.time() - start_time,2)} seconds")    
        print(f"Len of (test_data, train_data, features): ({train_num_samples}, {test_num_samples}, {num_features})")   

        
        # Reshape data:        
        meta_train = np.array(train_features_for_run[patient_info_columns])
        x_train = np.array(train_features_for_run.drop(columns=patient_info_columns))
        y_train = np.array(train_features_for_run["Group"])
            
        meta_test = np.array(test_features_for_run[patient_info_columns])
        x_test = np.array(test_features_for_run.drop(columns=patient_info_columns))
        y_test = np.array(test_features_for_run["Group"])
        
        # Standardize the data
        scaler = StandardScaler() 
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        svm_model = SVC(kernel=kernel_best, C=C_best, gamma=gamma_best) # Only use gamma for the rbf kernel 
        svm_model.fit(x_train, y_train)  # Train the model
        
        # Training  
        y_pred_train = svm_model.predict(x_train) 
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_accuracy = round(train_accuracy*100, 1)
        
        # Test 
        y_pred = svm_model.predict(x_test) # Test the model on the test set
        test_accuracy = accuracy_score(y_test, y_pred)
        test_accuracy = round(test_accuracy*100, 1)
        
        metadata_with_classification_result = add_correct_classified_column(y_true=y_test, y_pred=y_pred, metadata=meta_test, metadata_columns=patient_info_columns)
        
        metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'True class', y_test, True)
        metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'Predicted Class', y_pred, True)
        metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'Distance to decition boarder', np.round(svm_model.decision_function(x_test), 2),  True)
        
        roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set = calculate_classification_detailes_from_cross_fold(metadata_with_classification_result)

        
        # Generate the confusion matrix with labels
        confusion_mat = confusion_matrix(y_test, y_pred, labels=list(class_labels.values()))
        sensitivity, specificity = sensitivity_and_specificity(y_true=y_test, y_pred = y_pred, write_out=False)
        sensitivity = round(sensitivity*100, 1)
        specificity = round(specificity*100, 1)

                    
        # Add data to table: 
        results.loc[run_number, ["Feature", "numF", "Data subset", "Train num samples", "Test num samples", "TrainAcc", "Acc", "Sens", "Spec", "Roc Auc", "Distance to boarder for correct classified", "Distance to boarder for Wrongly classified", "TN", "TP", "FN", "FP", "Best threshold for test set"]] = [feature_name, num_features, train_data_name, train_num_samples, test_num_samples, train_accuracy, test_accuracy, sensitivity, specificity, roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set]
                     
        if num_data_subsets>1: # If multiple, also create a tablel with only the accuracy.     
            res_accuracies.at[feature_name, 'numF'] = num_features
            res_accuracies.at['num samples training', train_data_name] = train_num_samples
            res_accuracies.at['num samples test', train_data_name] = test_num_samples # Training and testing data name should be the same so use train_data_name here as well. 
            res_accuracies.at[feature_name, train_data_name] = test_accuracy
            
        # Printing out the results:         
        if print_out_each_run: # Print out some basic information for each run        
            print("Data over run: ")
            print(f"Fold Test Accuracy: {test_accuracy * 100:.2f}%")
            print("Confusion Matrix:")
            print(confusion_mat)
                
# If we want the data to be sorted by features instead of data. 
if sort_by_feature:
    results = results.sort_values(by=['Feature']) # Default is quicksort. If wanted stable sorting could use mergesort. 

# Print out table based on number of data types.    
print("---------------------------------------------------------------")
print("Table summary for all the runs: ")

printable_results = results[["Feature", "numF", "Data subset", "Train num samples", "Test num samples", "TrainAcc", "Acc", "Sens", "Spec", "Roc Auc"]]


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
max_row = results.loc[results['Acc'].idxmax()]
print(f"The best feature - data combination (based on avg Acc) among the ones tested is: {max_row['Feature']} and {max_row['Data subset']} with an an avg Acc on {max_row['Acc']}")

acc_sorted_results = results.sort_values(by=['Acc'], ascending=False)

# Save data to excell 
if save_to_excell:
    best_feature_data = pd.DataFrame({'Best feature': [max_row['Feature']], 'Best Data': [max_row['Data subset']], 'Acc': [max_row['Acc']]})
    parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'Train dataset', 'Test dataset', 'All data', 'seed_number', 'kernel_best', 'C_best', 'gamma_best', 'number_of_cross_val_folds', 'print_out_each_run',  'print_detailed_summary'], 'Value': ["svm model all GITA and EWA", train_dataset, test_dataset, def_experiment, seed_number, kernel_best, C_best, gamma_best, number_of_cross_val_folds, print_out_each_run, print_detailed_summary]})
    features_to_exel = pd.DataFrame(feature_subsets.items(), columns=['Subset', 'Features'])
    train_data_to_excel = pd.DataFrame(train_data_subset.items(), columns=['Train subset', 'Train data'])
    test_data_to_excel = pd.DataFrame(test_data_subset.items(), columns=['Test subset', 'Test data'])
    
    write_to_excel([res_accuracies, results, best_feature_data, acc_sorted_results, parameters_to_exel, features_to_exel, train_data_to_excel, test_data_to_excel], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell

#####################################################################

