
import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from functions import *
from constants import *
from definition_of_data_and_feature_groups import *
from sklearn.feature_selection import SequentialFeatureSelector as SFSsklearn

from mlxtend.feature_selection import SequentialFeatureSelector as SFSmlextend
from sklearn.feature_selection import RFE

from sklearn.model_selection import PredefinedSplit
import random

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  



from sklearn.pipeline import Pipeline


"""
Description of this file: 
--> Feature ranking with feature selection and random choises. 

--> GIVES OUT: 
    Gives one feature set for with an ideal number of features.
    Iterates through all features and find best acc for each number of features. 
    --> Creates a plot of accuracy vs feature amount. 
    --> Returns dataframe with accuracies and feature subset for all feature amounts.  
    
- If run type = "random" --> Randomly adds another feature to the set for each time, and creates a plot of feature amount vs accuracy. 

- If run type = "sklearn-SFS" --> Uses SequentialFeatureSelector from sklearn.
    --> See rather mlxtend-SFS. It is better for most use cases due to giving more information. 
    --> Can change between forward and backward method. 
    ---> https://scikit-learn.org/dev/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector 
    ---> Uses self defined cv and pipeline to get scaling for the differnt cv folds. 

- If run type = "mlxtend-SFS" --> Uses SequentialFeatureSelector from mlxtend. In this case we can also change between forward and backward method. 
     --> Can change between forward and backward method. 
    ---> Uses self defined cv and pipeline to get scaling for the differnt cv folds. 

- If run type = "RFE" --> Uses RFE from sklearn. 
"""



###########   DEFINING PARAMETERS FOR THE RUN   #####################

feature_sheet_name = "GITA-all" # "GITA-all" # "EWA-100" # EWA-balanced-69 # Sheet name we want to get features from. 
dataset = "GITA" # Eiter GITA or EWA.
fold_info_file = "kfold-groups-tsv.csv" # ewa, tsv or mf

filename_for_results = "SFS-sklearn-allDandF_100_samples" # Filename to save plot showing accuracy vs number of features.  
create_plots = True

run_type = "sklearn-SFS" # Either Random or sklearn-SFS or mlxtend-SFS or RFE

direction = "forward" # "forward" or "backward" # Only needed to be defined for run types of SFS 

number_of_features = 100 # Number of features wanted in the feature set. 

save_to_excell = True # If true it writes to Automated_results excell file


ids_for_test = ["A0042", "A0027", "A0020", "A0016", "A0009", "A0025", "A0007", "A0023", "A0006", "A0022", "AC0037", "AC0016", "AC0005", "AC0013", "AC0031", "AC0004", "AC0026", "AC0008", "AC0027", "AC0015"]



#######  PARAMETERS THAT CAN BE CHANGED BUT NOT NEEDED FOR MOST USE CASES #######

# Features
choosen_features = all_features # all_features # Define all features 

# Utterances 
utterance_type = [""] # [""] # ["Vowels", "modulated_vowels"] or etc.  For ex:["phonation"] for EWA. 
specific_utterance = [""] # ["a", "viaje"] or etc. 

# Seed number 
seed_number = 42

# SVM parameters --> Choosen based on grid search from grid_search
kernel_best ='rbf' 
C_best = 100
gamma_best = 0.0001

number_of_cross_val_folds = 10  # Need to change the fold info if something else than 10 is wanted.

#####################################################################


###########   CODE   ################################################

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
             
else: 
    print("No aproved dataset is defined. Choose either GITA or EWA")

fold_info_path = os.path.join(personal_path_to_balanced_folds,fold_info_file)  # Path to fold distribution information. 
fold_info = pd.read_csv(fold_info_path)

metadata_columns =  ["SEX", "AGE"] 
fold_info_columns = ["Fold"]
feature_info_columns = ["ID", "Utterance", "Utterance type", "Group"]

# patient_info_columns will be used later on to remove all metadata and then also want to remove id. 
patient_info_columns = feature_info_columns + metadata_columns + fold_info_columns

# Add metadata and fold info: 
features = add_columns_to_dataframe(features, metadata, metadata_columns) # Only bring along SEX and AGE from metadata. 
features = add_columns_to_dataframe(features, fold_info, fold_info_columns) # Only bring along Fold numhber from fold_info. 

features = remove_NaN(features, patient_info_columns, print_out = False) # Changes NaN values to be zero values instead. 

if utterance_type != [""]: # If defined utterance sub groups (like vowels or words or combinations)
        features = features[features["Utterance type"].isin(utterance_type)]

if specific_utterance != [""]: # If defined specific utterances (like "a" or "viaje" or combinations)
        features = features[features["Utterance"].isin(specific_utterance)]

features = features.loc[:, choosen_features + patient_info_columns]

np.random.seed(seed_number) 

run_number = 0
total_runs = len(choosen_features)
results = pd.DataFrame()
num_samples = features.shape[0]

if num_samples <= 0 or total_runs <= 0:
    raise ValueError("Not enough data or features are provided. Script stopped. Redefine feature and data subsets.")


num_folds = len(features["Fold"].unique())
    
plot_results = pd.DataFrame()

# Construct new test set and create predefined split for validation. 
test_data = features[features['ID'].isin(ids_for_test)]
train_data = features[~features['ID'].isin(ids_for_test)]

x_test = np.array(test_data.drop(columns=patient_info_columns))
y_test = np.array(test_data["Group"])

x_train = np.array(train_data.drop(columns=patient_info_columns))
y_train = np.array(train_data["Group"])

train_folds = np.array(train_data['Fold']) 
ps = PredefinedSplit(train_folds)

# Standardize the data
scaler = StandardScaler() 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

if run_type == "sklearn-SFS": 
    svm_model = SVC(kernel=kernel_best, C=C_best, gamma=gamma_best, random_state=seed_number) 
    pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", svm_model),
    ])   
    selected_features_previus = []
    new_features_added = []

    total_runs = number_of_features
    for n in tqdm(range(total_runs, total_runs + 1), total=total_runs, desc="Run for each feature"):
    
        selector = SFSsklearn(pipeline, n_features_to_select=n, direction=direction, cv = ps, scoring="accuracy") # Initialize SFE
        selector.fit(x_train, y_train) # Fit SFS on your data

        # Get the selected features
        
        selected_features = [feature for feature, support in zip(choosen_features, selector.support_) if support]
        
        x_train_temp = selector.transform(x_train)
        x_test_temp = selector.transform(x_test)
        
        scaler = StandardScaler() 
        x_train_temp = scaler.fit_transform(x_train_temp)
        x_test_temp = scaler.transform(x_test_temp)
        svm_model.fit(x_train_temp, y_train)
    
        # Training  
        y_pred_train = svm_model.predict(x_train_temp)  # reduces X to the selected features and predict using the estimator
        train_accuracy = round(accuracy_score(y_train, y_pred_train)*100, 2)
        
        # Test 
        y_pred = svm_model.predict(x_test_temp) # reduces X to the selected features and predict using the estimator
        test_accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
        
        results.loc[n, ["Selected number of features", "TrainAcc", "Acc",  "Selected feature list"]] = [selector.n_features_to_select_, train_accuracy, test_accuracy, str(selected_features)]

        new_features_added.append(list(set(selected_features)-set(selected_features_previus))) 
        
        selected_features_previus = selected_features

    if create_plots:
        image_path = os.path.join(personal_path_to_results, filename_for_results + generate_timestamp())
        create_feature_vs_acc_plot(results, image_path)

    acc_sorted_results = results.sort_values(by = ['Acc', "Selected number of features"], ascending = [False, True])  # Want to maximise Acc and minimize number of features (less likely to overtrain). Therefore sort by acc decending and number of features ascending. 
        
    diff_bet_best_worst = round(np.abs(np.array(acc_sorted_results["Acc"])[0]-np.array(acc_sorted_results["Acc"])[-1]), 2)
    
    plot_results.loc["Values for plotting", ["x (features)", "y (acc)", "y_train (train acc)", "x_max", "y_max"]] = [str(list(results["Selected number of features"])),str(list(results["Acc"])), str(list(results["TrainAcc"])), np.array(acc_sorted_results["Selected number of features"])[0], np.array(acc_sorted_results["Acc"])[0] ]

    display(results)
    
    if save_to_excell:

        parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'Run type', 'Feature amount', 'Direction', 'Dataset', 'num_samples', 'Utterance type', 'Specific utterance','seed_number', 'C_best', 'Gamma best', 'Kernel best', 'number_of_cross_val_folds', 'Feature sheet name', 'Features', 'Fold info file name', 'Selected next feature (from start to finish)', 'Difference between best and worst'],
                                        'Value': ["Best features seperate test set", run_type, total_runs, direction, dataset, num_samples, utterance_type, specific_utterance, seed_number, C_best, gamma_best, kernel_best, number_of_cross_val_folds, feature_sheet_name, list(features.keys()), fold_info_file, str(new_features_added), diff_bet_best_worst]})
        write_to_excel([results, parameters_to_exel, plot_results, acc_sorted_results], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell



# mlxtend SFS
elif run_type == "mlxtend-SFS":
    svm_model = SVC(kernel=kernel_best, C=C_best, gamma=gamma_best, random_state=seed_number) 
    pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", svm_model),
    ])   
    selected_features_previus = []
    new_features_added = []
    
    total_runs = number_of_features
    
    forward = True if direction =="forward" else False
    selector = SFSmlextend(pipeline, k_features="best", floating=False, verbose=2, forward=forward, cv = ps, scoring="accuracy") # Initialize SFE
    
    selector.fit(x_train, y_train) # Fit SFS on your data

    for n in tqdm(range(1, total_runs + 1) if direction == "forward" else range(total_runs, len(choosen_features)+1), total=total_runs, desc="Predicting for each feature amount"):
        selected_features_indexes = list(selector.subsets_[n]['feature_idx'])
        selected_features = [choosen_features[idx] for idx in selected_features_indexes]
        
        x_test_temp = test_data.loc[:, selected_features]
        x_train_temp = train_data.loc[:, selected_features]
        
        scaler = StandardScaler() 
        x_train_temp = scaler.fit_transform(x_train_temp)
        x_test_temp = scaler.transform(x_test_temp)
        svm_model.fit(x_train_temp, y_train)

        # Training  
        y_pred_train = svm_model.predict(x_train_temp)  # reduces X to the selected features and predict using the estimator
        train_accuracy = round(accuracy_score(y_train, y_pred_train)*100, 2)
        
        # Test 
        y_pred = svm_model.predict(x_test_temp)
        test_accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
        
        results.loc[n, ["Selected number of features", "TrainAcc", "Acc", "cv_scores", "avg cross val score",  "Selected feature list"]] = [n, train_accuracy, test_accuracy, str(selector.subsets_[n]['cv_scores']), round(selector.subsets_[n]['avg_score']*100,2), str(selected_features)]

        new_features_added.append(list(set(selected_features)-set(selected_features_previus))) 
        selected_features_previus = selected_features
        
        
    if create_plots:
        image_path = os.path.join(personal_path_to_results, filename_for_results + generate_timestamp())
        create_feature_vs_acc_plot(results, image_path)

    acc_sorted_results = results.sort_values(by = ['Acc', "Selected number of features"], ascending = [False, True])  # Want to maximise Acc and minimize number of features (less likely to overtrain). Therefore sort by acc decending and number of features ascending. 
        
    diff_bet_best_worst = round(np.abs(np.array(acc_sorted_results["Acc"])[0]-np.array(acc_sorted_results["Acc"])[-1]), 2)
    
    plot_results.loc["Values for plotting", ["x (features)", "y (acc)", "y_train (train acc)", "x_max", "y_max"]] = [str(list(results["Selected number of features"])),str(list(results["Acc"])), str(list(results["TrainAcc"])), np.array(acc_sorted_results["Selected number of features"])[0], np.array(acc_sorted_results["Acc"])[0] ]

    display(results)
    
    
    
    # For the best case: 
    selected_features_indexes = selector.k_feature_idx_
    selected_features = [choosen_features[idx] for idx in selected_features_indexes]
        
    # Get the selected features
        
    x_train_temp = selector.transform(x_train)
    x_test_temp = selector.transform(x_test)
    
    scaler = StandardScaler() 
    x_train_temp = scaler.fit_transform(x_train_temp)
    x_test_temp = scaler.transform(x_test_temp)
    svm_model.fit(x_train_temp, y_train)

    y_pred_train = svm_model.predict(x_train_temp)  # reduces X to the selected features and predict using the estimator
    train_accuracy = round(accuracy_score(y_train, y_pred_train)*100, 2)
    
    y_pred = svm_model.predict(x_test_temp) # reduces X to the selected features and predict using the estimator
    test_accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
        
    best_res = pd.DataFrame(({'Best selected feature subset': ["Selected number of features", "TrainAcc", "Acc", "Avg cross val acc", "Selected feature list"], "Values": [len(x_train_temp[0]), train_accuracy, test_accuracy, round(selector.k_score_*100,2), str(selected_features)]}))
    
    if save_to_excell:

        parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'Run type', 'Feature amount', 'Direction', 'Dataset', 'num_samples', 'Utterance type', 'Specific utterance','seed_number', 'C_best', 'Gamma best', 'Kernel best', 'number_of_cross_val_folds', 'Feature sheet name', 'Features', 'Fold info file name', 'Selected next feature (from start to finish)', 'Difference between best and worst'],
                                        'Value': ["Best features seperate test set", run_type, total_runs, direction, dataset, num_samples, utterance_type, specific_utterance, seed_number, C_best, gamma_best, kernel_best, number_of_cross_val_folds, feature_sheet_name, list(features.keys()), fold_info_file, str(new_features_added), diff_bet_best_worst]})
        write_to_excel([results, best_res, parameters_to_exel, plot_results, acc_sorted_results], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell



elif run_type == "Random":
    svm_model = SVC(kernel=kernel_best, C=C_best, gamma=gamma_best, random_state=seed_number) 

    selected_features = []
    
    updated_features = choosen_features
            
    for n in tqdm(range(1, 30 + 1), total=30, desc="Run for each feature"):

        feature_for_round = random.choice(updated_features) # Randomly choosen features. 
        selected_features.append(feature_for_round)
        updated_features.remove(feature_for_round)
        
        x_test_temp = test_data.loc[:, selected_features]
        x_train_temp = train_data.loc[:, selected_features]
        
        scaler = StandardScaler() 
        x_train_temp = scaler.fit_transform(x_train_temp)
        x_test_temp = scaler.transform(x_test_temp)
        svm_model.fit(x_train_temp, y_train)
    
        # Training  
        y_pred_train = svm_model.predict(x_train_temp)  # reduces X to the selected features and predict using the estimator
        train_accuracy = round(accuracy_score(y_train, y_pred_train)*100, 2)
        
        # Test 
        y_pred = svm_model.predict(x_test_temp) # reduces X to the selected features and predict using the estimator
        test_accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
        
        results.loc[n, ["Selected number of features", "TrainAcc", "Acc",  "Selected feature list"]] = [len(selected_features), train_accuracy, test_accuracy, str(selected_features)]
                

    if create_plots:
        image_path = os.path.join(personal_path_to_results, filename_for_results + generate_timestamp())
        create_feature_vs_acc_plot(results, image_path)

    acc_sorted_results = results.sort_values(by = ['Acc', "Selected number of features"], ascending = [False, True])  # Want to maximise Acc and minimize number of features (less likely to overtrain). Therefore sort by acc decending and number of features ascending. 
        
    diff_bet_best_worst = round(np.abs(np.array(acc_sorted_results["Acc"])[0]-np.array(acc_sorted_results["Acc"])[-1]), 2)
    
    plot_results.loc["Values for plotting: ", ["x (features)", "y (acc)", "y_train (train acc)", "x_max", "y_max"]] = [str(list(results["Selected number of features"])),str(list(results["Acc"])), str(list(results["TrainAcc"])), np.array(acc_sorted_results["Selected number of features"])[0], np.array(acc_sorted_results["Acc"])[0] ]

    display(results)
    
    if save_to_excell:

        parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'Run type', 'Feature amount', 'Direction', 'Dataset', 'num_samples', 'Utterance type', 'Specific utterance','seed_number', 'C_best', 'Gamma best', 'Kernel best', 'number_of_cross_val_folds', 'Feature sheet name', 'Features', 'Fold info file name', 'Selected next feature (from start to finish)', 'Difference between best and worst'],
                                        'Value': ["Best features seperate test set", run_type, total_runs, direction, dataset, num_samples, utterance_type, specific_utterance, seed_number, C_best, gamma_best, kernel_best, number_of_cross_val_folds, feature_sheet_name, list(features.keys()), fold_info_file, str(selected_features), diff_bet_best_worst]})
        write_to_excel([results, parameters_to_exel, plot_results, acc_sorted_results], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell




elif run_type == "RFE":
    C_linear = 0.01
    svm_model = SVC(kernel="linear", C=C_linear, random_state=seed_number) 

    selected_features_previus = []
    new_features_added = []

    total_runs = number_of_features
    for n in tqdm(range(1, total_runs + 1), total=total_runs, desc="Run for each feature"):
    
        selector = RFE(svm_model, n_features_to_select=n, step=1, verbose=0) 
        selector.fit(x_train, y_train) # Fit SFS on your data

        # Get the selected features
        selected_features = [feature for feature, support in zip(choosen_features, selector.support_) if support]
    
        # Training  
        y_pred_train = selector.predict(x_train)  # reduces X to the selected features and predict using the estimator
        train_accuracy = round(accuracy_score(y_train, y_pred_train)*100, 2)
        
        # Test 
        y_pred = selector.predict(x_test) # reduces X to the selected features and predict using the estimator
        test_accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
        
        results.loc[n, ["Selected number of features", "TrainAcc", "Acc",  "Selected feature list"]] = [selector.n_features_, train_accuracy, test_accuracy, str(selected_features)]

        new_features_added.append(list(set(selected_features)-set(selected_features_previus))) 
        
        selected_features_previus = selected_features

    if create_plots:
        image_path = os.path.join(personal_path_to_results, filename_for_results + generate_timestamp())
        create_feature_vs_acc_plot(results, image_path)

    acc_sorted_results = results.sort_values(by = ['Acc', "Selected number of features"], ascending = [False, True])  # Want to maximise Acc and minimize number of features (less likely to overtrain). Therefore sort by acc decending and number of features ascending. 
        
    diff_bet_best_worst = round(np.abs(np.array(acc_sorted_results["Acc"])[0]-np.array(acc_sorted_results["Acc"])[-1]), 2)
    
    plot_results.loc["Values for plotting", ["x (features)", "y (acc)", "y_train (train acc)", "x_max", "y_max"]] = [str(list(results["Selected number of features"])),str(list(results["Acc"])), str(list(results["TrainAcc"])), np.array(acc_sorted_results["Selected number of features"])[0], np.array(acc_sorted_results["Acc"])[0] ]

    display(results)
    
    if save_to_excell:

        parameters_to_exel = pd.DataFrame({'Parameters': ['Python file', 'Run type', 'Feature amount', 'Dataset', 'num_samples', 'Utterance type', 'Specific utterance','seed_number', 'C_best', 'Kernel best', 'number_of_cross_val_folds', 'Feature sheet name', 'Features', 'Fold info file name', 'Selected next feature (from start to finish)', 'Difference between best and worst'],
                                        'Value': ["Best features seperate test set", run_type, total_runs, dataset, num_samples, utterance_type, specific_utterance, seed_number, C_linear, "linear", number_of_cross_val_folds, feature_sheet_name, list(features.keys()), fold_info_file, str(new_features_added), diff_bet_best_worst]})
        write_to_excel([results, parameters_to_exel, plot_results, acc_sorted_results], personal_path_to_code + "/Automated_results.xlsx") # results_to_exell
        
else:
    ValueError("Wrong initianlisation of run type. Redefine to Random or SFS and run again. ")