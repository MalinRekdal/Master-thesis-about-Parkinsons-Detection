
"""
SVM grid search
Used to find the optimal parameters for SVM model. 
"""

# Imports

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline


from functions import *
from constants import *
from definition_of_data_and_feature_groups import *

###########   DEFINING PARAMETERS FOR THE RUN   #################

# Seed number 
seed_number = 42
np.random.seed(seed_number) 

features_for_model = all_features # all_features  # all_features # ["avg BBEon_1"] # phonation_all_features #  all_features, phonation_all_features, etc. 

# Utterances 
utterance_type = [""] # ["Vowels", "modulated_vowels"] or etc. 
specific_utterance = [""] # ["a", "viaje"] or etc. 

dataset = "EWA" # Eiter GITA or EWA.
feature_sheet_name = "EWA-100" # "GITA-all" # "EWA-100" # EWA-balanced-69 # Sheet name we want to get features from. 
fold_file = "kfold-groups-ewa.csv" #tsv, mf or ewa. 


# Grid search to experiment with different parameters, automatic does cross validation with 5 fold cross val so dont need a validation set. 
C_linear = [0.0001, 0.001, 0.01, 0.1, 1, 10]  # Takes a lot of time to fit the linear model with higher C values so we restrict it. 
C_rbf = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] 
gamma = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] 

save_to_excell = True
all_folds = True  #  Defines if we want to run the grid search ones for every fold (True) or only once for one fold (False)
print_out_each_fold = False

# Only needed to be defined if all_folds = False
test_fold_numbers = [0]  # Define one or muliple folds to test for. Best for fold = 0, uses the 9 others for training. 


#################################################################

feature_path = personal_path_to_code + '/Features.xlsx'   # Defines path to feature folder to use. 
features = pd.read_excel(feature_path, sheet_name=feature_sheet_name, index_col=0)

if dataset == "GITA":
    metadata_path = os.path.join(personal_path_to_PC_GITA,'PCGITA_metadata.xlsx')
    metadata = pd.read_excel(metadata_path)
    metadata = restructure_id(metadata)
    
elif dataset == "EWA": 
    metadata_path = os.path.join(personal_path_to_EWA_DB,'SPEAKERS.TSV')
    metadata=pd.read_csv(metadata_path,sep='\t') # Get all metadata 
    
    metadata = metadata.rename(columns={"SPEAKER_CODE": "ID"}) # Restructure IDs to be called "ID". 
    metadata = metadata.loc[metadata['ID'].isin(list(features["ID"]))] # Extract the data with the IDs from the feature data. 

fold_info_path = os.path.join(personal_path_to_balanced_folds, fold_file) # File where we have the fold distribution saved. 
fold_info = pd.read_csv(fold_info_path)


metadata_columns =  ["SEX", "AGE"]
fold_info_columns = ["Fold"]
feature_info_columns = ["ID", "Utterance", "Utterance type", "Group"]

# patient_info_columns will be used later on to remove all metadata and then also want to remove id. 
patient_info_columns = feature_info_columns + metadata_columns + fold_info_columns

# Add metadata and fold info: 
features = add_columns_to_dataframe(features, metadata, metadata_columns)
features = add_columns_to_dataframe(features, fold_info, fold_info_columns)

# Remove NaN
features = remove_NaN(features, patient_info_columns, print_out = False)

# Sorting out the data and features we want to use: 
if utterance_type != [""]: # If defined utterance sub groups (like vowels or words or combinations)
        features = features[features["Utterance type"].isin(utterance_type)]

if specific_utterance != [""]: # If defined specific utterances (like "a" or "viaje" or combinations)
        features = features[features["Utterance"].isin(specific_utterance)]

if features_for_model:
        features = features.loc[:, features_for_model + patient_info_columns]


if all_folds: # # Grid search for all 10 folds: 
        num_folds = len(features["Fold"].unique())
        test_fold_numbers = "All data"
elif not all_folds: # # Grid search one fold 
        num_folds = len(test_fold_numbers)
else:
        print("Something went wrong, redefine your parameters for the run. ")


# Defining parameter grid: 

param_grid = [{'model__kernel': ['rbf'], 'model__gamma': gamma, 'model__C': C_rbf},  # RBF = Gaussian Radial Basis Function 
            {'model__kernel': ['linear'], 'model__C': C_linear}] 

grid_search_information = pd.DataFrame()
desc_description = "Grid search for one fold"  if num_folds == 1 else "Grid search for all folds"

all_fitting_data_df_list = []

for fold_num in tqdm(range(num_folds), total=num_folds, desc=desc_description): # Iterate through folds 
        if not all_folds: 
                fold_num = test_fold_numbers[fold_num]
                verbose = 3 # Print out more info when we look into fewer folds. 
        else:
                verbose = 1
        
        # Reshape data: 
        test_data = features[features['Fold'] == fold_num]

        x_test = np.array(test_data.drop(columns=patient_info_columns))
        y_test = np.array(test_data["Group"])

        train_data = features[features['Fold'] != fold_num]
        
        train_folds = np.array(train_data['Fold']) 
        x_train = np.array(train_data.drop(columns=patient_info_columns))
        y_train = np.array(train_data["Group"])

        # Standardize the data
        scaler = StandardScaler() 
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        ps = PredefinedSplit(train_folds)
        
        pipeline = Pipeline(steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(random_state=seed_number)),
        ])

        grid = GridSearchCV(pipeline, cv = ps, param_grid = param_grid, refit = True, verbose = verbose, return_train_score=True) 
                # verbose controls how much is printed out, refit = True makes grid be a svm model fitted to the best result from the grid search on all folds from the training set. 
                # The parameters selected are those that maximize the score of the left out data
        grid.fit(x_train, y_train)  
        
        all_mean_train_scores = np.round(grid.cv_results_["mean_train_score"]*100, 2)
        all_mean_test_scores = np.round(grid.cv_results_["mean_test_score"]*100, 2)
        all_params = grid.cv_results_["params"]
        
        # Create a DataFrame based on data from all the fits. 
        all_fitting_data_df = pd.DataFrame({
                "Test fold number": fold_num, 
                "Fit number": [i for i in range(1,len(all_mean_test_scores)+1)],
                "All params fitted for": all_params,
                "Avg train score for all fits": all_mean_train_scores,
                "Avg test score for all fits": all_mean_test_scores
        })
        all_fitting_data_df = all_fitting_data_df.sort_values(by=['Avg test score for all fits'], ascending=False)
        
        all_fitting_data_df_list.append(all_fitting_data_df)
        print(all_fitting_data_df_list)

        # Extract train and test scores for the specific parameter combination across all folds
        best_train_scores = [round(grid.cv_results_[f'split{i}_train_score'][grid.best_index_]*100, 2) for i in range(9)]
        best_test_scores = [round(grid.cv_results_[f'split{i}_test_score'][grid.best_index_]*100, 2) for i in range(9)]
        avg_score = round(grid.best_score_*100, 2)
        
        train_predictions = grid.predict(x_train)
        accuracy_train = round(accuracy_score(y_train, train_predictions)*100, 2)

        test_predictions = grid.predict(x_test)
        accuracy_test = round(accuracy_score(y_test, test_predictions)*100, 2)
        
        
        if print_out_each_fold:
                print(" ------------------------------------------------------------ ")
                print(f"Results for fold {fold_num} for the best parameters:")
                print(" ")
                print("For all the parameters: ")
                display(all_fitting_data_df)
                
                print(" ")
                print("Result for best parameters: ")
                print(f"\t The best parameters after grid search is {grid.best_params_}, and therefore now our SVM model: {grid.best_estimator_}") 
                print(f"\t The average score for this fit was {avg_score}")
                
                print(f"\t The {ps.get_n_splits()} folds train scores {best_train_scores}")
                print(f"\t The {ps.get_n_splits()} folds test scores {best_test_scores}")
                
                
                print("Result for best parameters after fitting the model and retraining: ")
                print(f"\n Training Accuracy: {accuracy_train}%")
                print(f"\n Test Accuracy: {accuracy_test}%")
                _ = sensitivity_and_specificity(y_true=y_test, y_pred = test_predictions) # Calculate and print out sensitivity and specificity
                calculate_confusion_matrix(y_true=y_test, y_pred = test_predictions) # Calculate and print out confusion matrix
                print_classification_report(y_true=y_test, y_pred=test_predictions) # Print out classification report
                
        
        grid_search_information.at[fold_num, "Fold number for test"] = int(fold_num)
        grid_search_information.at[fold_num, "Average validation score"] = avg_score 
        grid_search_information.at[fold_num, "Train accuracy on best fitted model"] = accuracy_train
        grid_search_information.at[fold_num, "Test accuracy on best fitted model"] = accuracy_test
        grid_search_information.at[fold_num, "Best grid params"] = str(grid.best_params_)
        grid_search_information.at[fold_num, f"Train scores for all {ps.get_n_splits()} folds"] = str(best_train_scores)
        grid_search_information.at[fold_num, f"Test scores for all {ps.get_n_splits()} folds "] = str(best_test_scores)

printable_results = grid_search_information[["Fold number for test", "Average validation score", "Train accuracy on best fitted model", "Test accuracy on best fitted model", "Best grid params"]]
display(printable_results)

if save_to_excell:
        parameters_to_excel = pd.DataFrame({'Parameters': ['Python file', 'Dataset', 'seed_number', 'grid searach params', 'Feature sheet name', 'Fold info file', 'Number of test folds', 'Test fold numbers', 'Number of validation folds'], 'Value': ["grid search", dataset, seed_number, param_grid, feature_sheet_name, fold_file, 10, test_fold_numbers, ps.get_n_splits()]})
        features_and_data_to_excel = pd.DataFrame({
                'Type': ['Features', 'Utterance type', 'Specific utterance'],
                'Value': [list(features.keys()), utterance_type, specific_utterance]
                })
        
        write_to_excel_list = [grid_search_information, parameters_to_excel, features_and_data_to_excel] + all_fitting_data_df_list
        write_to_excel(write_to_excel_list, personal_path_to_code + "/Automated_results.xlsx") # results_to_exell

