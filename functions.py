"""
This python file contains functions used in multiple files the rest of the project (mostly to extract the right data), and functions to evaluate a model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from constants import class_labels, correct_color, wrong_color, HC_color, PD_color
import pytz
from datetime import datetime 
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# Functions used that is reused in different files: 

def get_health_state(value):
    """
    Function to get health state from number 0 or 1 using the label mapping dict in the opposite way. 
    
    The functionality to get the key from the value from a dict is taken from: 
    https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/

    Args:
        value (int): The value 0 or 1 we want to get the health state for. 

    Returns:
        string: the string showing if it is HC or PD (the health state). 
    """
    key = list(filter(lambda x: class_labels[x] == value, class_labels))[0]
    return key


def extend_paths(path_list, base_path):
    """
    Extends all elements in path_list with the base_path first

    Args:
        path_list (list): list of paths 
        base_path (str): str of base path

    Returns:
        list: new list with all paths extended with the base path
    """
    for i in range(len(path_list)):
        if base_path not in path_list[i]:
            path_list[i] = base_path + path_list[i]
    return path_list
    

def find_certain_paths_from_all_paths(paths, d_type):
    """
    Takes inn a list of paths and a data types parameter and 
    returns all paths that contain that data types.
    If none of d_type is in the paths then it returns an warning message about that.  

    Args:
        paths (list): list of paths we want to reduce
        d_type (list): list of data types we want to keep

    Returns:
        list: new list of paths that only contain data type. 
    """
    res = [elem for elem in paths if any(term in elem for term in d_type)]
    if res != []:
        return res
    else:
        print("Could not find ", d_type, ". Check your definitions again.")
        return False


def add_columns_to_dataframe(to_dataframe, from_dataframe, columns_to_add):
    """
    Takes in a df of features and adds columns corresponding to the metadata we want to add. 

    Args:
        data (df): DataFrame to add metadata to
        metadata (DataFrame): DataFrame with the metadata we want to add. 
        metadata_columns (list): List of the metadata types we want to add

    Returns:
        df: data with added metadata
    """
    result = to_dataframe.copy()
    for elem in columns_to_add:
        result[elem] = ""
    for index, row in result.iterrows():
        for elem in columns_to_add:
            result.at[index, elem] = from_dataframe.loc[from_dataframe['ID'] == row["ID"], elem].iloc[0] # Add metadata info: 
    return result


def restructure_id(metadata):
    """
    Restructure id's in metadata to match the id's in the fold info. The ID's will therefore be called
    ID, and be in the form "A0013" or "AC0013" for ID 13. 
    
    Note that this is only needed to be done with the metadata because fold_info and features info already is on an ideal form, 
    namely ["ID"] A0013. Feature information is created in the com_create_features_from_wav.py file to be ideal. 

    Args:
    metadata (DataFrame): DataFrame with the metadata and an id column called "RECODING ORIGINAL NAME". 
    
    Returns:
        metadata after ID columns are restructured. 
    """
    
    # METADATA --> Original: ["RECODING ORIGINAL NAME"] AVPEPUDEA0005   
    id_list_metadata=list(metadata['RECODING ORIGINAL NAME'])
    for id in range(len(id_list_metadata)):
        id_list_metadata[id]=id_list_metadata[id].replace('AVPEPUDE','')
    metadata.insert(0,'ID',id_list_metadata, True)
    metadata.drop(columns=["RECODING ORIGINAL NAME"], inplace=True)
    return metadata


def remove_NaN(features, metadata_columns, print_out = True):
    """
    Removes NaN values by changing them to 0 values, and evaluates how many values there is. 
    
    Args: 
    features (DataFrame): DataFrame with features and metadata. 
    metadata_columns (list): list of the types of metadata we have
    print_out (bool, optional): Variable to decide if you want NaN information printed or not. Defaults to True. 
    
    Returns: 
    features (DataFrame): Feature DataFrame where all NaN values is changed to 0 values.  
    """
    if print_out: 
        features_without_metadata = features.drop(columns=metadata_columns)
        print("Number of Zero values:", (features_without_metadata == 0).sum().sum())
        print("Number of NaN values:", features_without_metadata.isna().sum().sum())
        print("Total number of values from features: ", features_without_metadata.size)

    
    features.infer_objects(copy=False).fillna(0, inplace=True) # Deprecated method: # features.fillna(0, inplace=True) 
    
    return features



########## EVALUATION FUNCTIONS ##################

def plot_histogram(data, xlabel, title, ylim=False, color='skyblue', bins=False):
    """
    Plots histogram for data. 

    Args:
        data (list): list of data to plot histogram of. 
        xlabel (str): x label
        title (str): title
        ylim (tuple or bool): If wanted, define ylim, if not the automated ylim will be set.
        color (str, optional): Color to plot histogram in. Defaults to 'skyblue'.
        bins (bool, optional): bins we want the histogram to be devided into. Defaults to False, and if False then plt.hist uses its natural choice.
    """
    if bins:
        data.plot(kind='hist', color=color, edgecolor='black', bins = bins)
    else: 
        data.plot(kind='hist', color=color, edgecolor='black')
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Number of people', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if ylim:
        plt.ylim(ylim)
    plt.show()
    
    
def plot_bar(labels, values, max, title, xlabel, ylabel, fold):
    """
    Plots a bar plot based on input information. 
    
    Args:
    labels (list): List of labels to put on the x axis
    values (list): List of values for the corresponding x axis values. 
    max (int): Indicates the number a bad maximum can have. 
    title (string): Title for the plot. 
    xlabel (string): label for the x axis. 
    ylabel (string): label for the y axis.
    fold (bool, default = True): fold indicates if the evaluation is done on folds, or on the whole data. 
    """
    colors = [HC_color if "AC" in label else PD_color for label in labels]
    
    if fold:
        plt.xticks(rotation=45)
    else: 
        plt.figure(figsize=(15, 4.8))
        plt.xticks(rotation=90)
    
    
    plt.bar(labels, values, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.axhline(y=np.mean(values), color='black', linestyle='--', label='Average amount of correct utterances')
    plt.axhline(y=max/2, color='black', linestyle=':', label='Half of the utterances')
    plt.legend()
    plt.show()


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculates and prints confusion matrix for both number of instances and percentage, and
    plots confusion matrix for number of instances. Calculates from true and predicted labels. 

    Args:
        y_pred (list): list of predicted labels
        test_labels (list): list of true labels
    """
    # Generate the confusion matrix with labels
    confusion_mat = confusion_matrix(y_true, y_pred, labels=list(class_labels.values()))

    # Display the confusion matrix: 
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=list(class_labels.keys()))
    disp.plot()
    plt.title("Confusion matrix")
    plt.show()
    
    
def print_classification_report(y_true, y_pred):
    """
    Creates and prints out more detailed classification report from true and predicted labels. 

    Args:
        test_labels (list): list of true labels
        y_pred (list): list of predicted labels
    """
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(class_labels.keys())))
    

def sensitivity_and_specificity(y_true, y_pred, write_out=True):
    """
    Uses true labels and predicted labels to find sensitivity and specificity.
    Formulas gotten from: https://www.analyticsvidhya.com/blog/2021/06/classification-problem-relation-between-sensitivity-specificity-and-accuracy/ 

    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        write_out (bool, optional): Variable to decide if you want sensitivity and specificity values printed or not. Defaults to True.

    """
    confusion_mat = confusion_matrix(y_true, y_pred, labels=list(class_labels.values()))
    TN = confusion_mat[0, 0] # True Negative (True: HC, Pred: HC)
    TP = confusion_mat[1, 1] # True Positive (True: PD, Pred: PD)
    FN = confusion_mat[1, 0] # False Negative (True: PD, Pred: HC)
    FP = confusion_mat[0, 1] # False Positive (True: HC, Pred: PD)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    if write_out:
        print(f"Sensitivity (Recall of PD): {sensitivity* 100:.2f}%")
        print(f"Specificity (Recall of HC): {specificity* 100:.2f}%")
    return sensitivity, specificity


def autolabel(rects, ax, size=16):
    """
    Function that ads numbers above the bars in the gender distribution plot
    Note: This function is created with generative AI (Chat GPT) and then modified for this use. 
    Args:
        rects (bars): The bars for male and female for either correct or wrongly classified
        ax (ax): the axis
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    size=size)




########## EVALUATE METADATA FUNCTIONS ##################


def add_correct_classified_column(y_true, y_pred, metadata, metadata_columns):
    """
    Takes in metadata and adds a column informing if the data was correct (True) or wrongly (False) classified. 

    Args:
        y_true (list): list of true labels
        y_pred (list): list of predicted labels
        metadata (DataFrame): DataFrame to add column of correctly classified information to. 
        metadata_columns (list): list of corresponding metadata to the true and predicted labels
    
    Returns metadata with a new column indicating if the sample was correct (True) or wrongly (False) classified.
    """
    
    data = pd.DataFrame(metadata, columns=metadata_columns)
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        corr_classified = true == pred
        data.at[i, "Correctly classified"] = corr_classified
    
    return data




########## CROSS FOLD FUNCTIONS ##################



def balanced_cross_validation(data, model, metadata_columns, write_out_each_fold=True, test_data=pd.DataFrame(), probability=False):
    """
    Does a 10 fold cross validation using data and the labels and already defined model. 
        
    Args:
        data (dataframe): all data, metadata and labels
        model (model): The trained model we want to predict from.
        metadata_columns (dataframe): List of metadata columns that are in data. 
        write_out_each_fold (bool, optional): Variable to decide if you want fold values and conf matrix printed or not. Defaults to True.      
        test_data (dataframe, optional): Seperate defined test dataset if we want to test on another dataset than we train on. Each fold will be tested on all the data. Defaults to False. 
        probability (bool, optional): Bool variable to define if the svm_model is defined to work with probabilities or not. Defaults to False. 
        
    returns list of accuracy scores, training accuracy scores, sensitivity scores and specificity scores for the 10 folds, as well as the confusion matrix sum over the 10 folds. 
    """
    # List to store accuracy, sensitivity and specificity scores for each fold
    accuracy_scores = []
    training_accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    confusion_mat_sum = np.array([[0, 0], [0, 0]]) # To store confusion matrix sum from all folds
    metadata_with_classification_result = pd.DataFrame()
    distance_to_decition_boarder = []
    probabilities_HC = []
    probabilities_PD = []
    all_y_test = []
    all_y_pred = []
    
    num_folds = len(data["Fold"].unique())

    
    for fold_num in tqdm(range(num_folds), total=num_folds, desc="Cross validation progress"): # Iterate through folds 
        if write_out_each_fold:
            print(" ")
            print(f"This is data for fold number {fold_num}: ")
            
        # Reshape data:         
        fold_train_data = data[data['Fold'] != fold_num]
        
        x_train = np.array(fold_train_data.drop(columns=metadata_columns))
        y_train = np.array(fold_train_data["Group"])
        
        if test_data.empty: # If train data and test data is from the same dataset 
            fold_test_data = data[data['Fold'] == fold_num]
        else: # If test data is seperatly defined and is form a seperate dataset. 
            fold_test_data = test_data             
        
        meta_test = np.array(fold_test_data[metadata_columns])
        x_test = np.array(fold_test_data.drop(columns=metadata_columns))
        y_test = np.array(fold_test_data["Group"])
        
        # Standardize the data
        scaler = StandardScaler() 
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model.fit(x_train, y_train)  # Train the model
        
        # Training  
        y_pred_train = model.predict(x_train) 
        train_accuracy = accuracy_score(y_train, y_pred_train)
        training_accuracy_scores.append(train_accuracy) # Append train accuracy scores
        
        # Test 
        y_pred = model.predict(x_test) # Test the model on the test set
        test_accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(test_accuracy) # Append accuracy scores
        
        metadata_with_classification_result = pd.concat([metadata_with_classification_result, add_correct_classified_column(y_true=y_test, y_pred=y_pred, metadata=meta_test, metadata_columns=metadata_columns)]) 
        
        all_y_test = np.concatenate((all_y_test, y_test)) 
        all_y_pred = np.concatenate((all_y_pred, y_pred)) 
        
        distance_to_decition_boarder = np.concatenate((distance_to_decition_boarder, model.decision_function(x_test)))  # Shows distance to dectition boundery. 
        if probability: # Shows probability for each of the classes. Only in use if probability is True and we have the probability information from the model. 
            probabilities_HC = np.concatenate((probabilities_HC, model.predict_proba(x_test)[:,0]))
            probabilities_PD = np.concatenate((probabilities_PD, model.predict_proba(x_test)[:,1])) 
        
        # Generate the confusion matrix with labels
        confusion_mat = confusion_matrix(y_test, y_pred, labels=list(class_labels.values()))
        confusion_mat_sum += confusion_mat     
        
        sensitivity, specificity = sensitivity_and_specificity(y_true=y_test, y_pred = y_pred, write_out=False)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)  
        
        # Print result for fold: 
        if write_out_each_fold:
            print(f"Fold Test Accuracy: {test_accuracy * 100:.2f}%")
            print("Confusion Matrix:")
            print(confusion_mat)
            
    
    metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'True class', all_y_test, True)
    metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'Predicted Class', all_y_pred, True)

    metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'Distance to decition boarder', np.round(distance_to_decition_boarder, 2),  True)
    if probability:
        metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'Probabilities for HC', np.round(probabilities_HC, 2), True)
        metadata_with_classification_result.insert(len(metadata_with_classification_result.columns),'Probabilities for PD', np.round(probabilities_PD, 2), True)

    return accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores, confusion_mat_sum, metadata_with_classification_result



def write_out_cross_fold_results(num_folds, accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores, confusion_mat_sum):
    """
    Writes out cross fold validation results in a pretty way. 
    
    Args: 
        num_folds (int): Number of folds used in the cross fold validation. 
        accuracy_scores (list): List of accuracy scores from the num_folds folds. 
        training_accuracy_scores (list): List of training accuracy scores from the num_folds folds. 
        sensitivity_scores (list): List of sensitivity scores from the num_folds folds. 
        specificity_scores (list): List of specificity scores from the num_folds folds. 
        confusion_mat_sum (array): Matrix showing the confusion matrix sum over the num_folds folds. 
    """
    _, _, _, avgAcc, avgTrainAcc, avgSens, avgSpec = calculate_validation_measures_from_cross_fold(accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores)
    
    print(f"Results over all {num_folds} folds:")
    rounded_accuracy_scores = [round(elem * 100, 2) for elem in accuracy_scores]
    print(" ")
    print(f"Accuracy over all {num_folds} folds: {rounded_accuracy_scores}")
    print(f"Average Accuracy: {avgAcc}%")
    print(f"The accuracy is variating between {np.min(rounded_accuracy_scores)}% and {np.max(rounded_accuracy_scores)}%")
    
    rounded_train_accuracy_scores = [round(elem * 100, 2) for elem in training_accuracy_scores]
    print(" ")
    print(f"Training accuracy over all {num_folds} folds: {rounded_train_accuracy_scores}")
    print(f"Average Training Accuracy: {avgTrainAcc}%")
    print(f"The Training accuracy is variating between {np.min(rounded_train_accuracy_scores)}% and {np.max(rounded_train_accuracy_scores)}%")
    
    rounded_sensitivity_scores = [round(elem * 100, 2) for elem in sensitivity_scores]
    print(" ")
    print(f"Sensitivity over all {num_folds} folds: {rounded_sensitivity_scores}")
    print(f"Average Sensitivity: {avgSens}%")
    print(f"The Sensitivity is variating between {np.min(rounded_sensitivity_scores)}% and {np.max(rounded_sensitivity_scores)}%")
    
    rounded_specificity_scores = [round(elem * 100, 2) for elem in specificity_scores]
    print(" ")
    print(f"Specificity over all {num_folds} folds: {rounded_specificity_scores}")
    print(f"Average Specificity: {avgSpec}%")
    print(f"The Specificity is variating between {np.min(rounded_specificity_scores)}% and {np.max(rounded_specificity_scores)}%")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat_sum, display_labels=list(class_labels.keys()))
    disp.plot()
    plt.title(f"Confusion matrix sum over all {num_folds} folds:")
    
    print("Sum of conf matrixes:")
    print(confusion_mat_sum)


def calculate_validation_measures_from_cross_fold(accuracy_scores, training_accuracy_scores, sensitivity_scores, specificity_scores):
    """
    Calculates average and std of Accuracy, training accuracy, sensitivity and specificity among the 10 folds from the cross validation based on the args and 
    returns it as a string: avg u"\u00B1" std (avg +- std). 
    
    Args: 
        accuracy_scores (list): List of accuracy scores from the num_folds folds. 
        training_accuracy_scores (list): List of training accuracy scores from the num_folds folds. 
        sensitivity_scores (list): List of sensitivity scores from the num_folds folds. 
        specificity_scores (list): List of specificity scores from the num_folds folds.
        
    Returns: average Accuracy, std of acc and avg train acc as a value as well as average accuracy, average training accuracy, average sensitivity and average specificity with corresponding std values as a string. 
    """
    avgAcc = round(np.mean(accuracy_scores)*100, 1)
    avgTrainAcc = round(np.mean(training_accuracy_scores)*100, 1)
    avgSens = round(np.mean(sensitivity_scores)*100, 1)
    avgSpec = round(np.mean(specificity_scores)*100, 1)
    
    
    # FIX: Calculate all the std values: 
    stdAcc = round(np.std(accuracy_scores)*100, 1)
    stdTrainAcc = round(np.std(training_accuracy_scores)*100, 1)
    stdSens = round(np.std(sensitivity_scores)*100, 1)
    stdSpec = round(np.std(specificity_scores)*100, 1)
    
    pm = u" \u00B1 " # u"\u00B1" gives +- as a pm sign where + is above -.
    str_acc = str(avgAcc) + pm + str(stdAcc)
    str_TrainAcc = str(avgTrainAcc) + pm + str(stdTrainAcc)
    str_sens = str(avgSens) + pm + str(stdSens)
    str_spec = str(avgSpec) + pm + str(stdSpec)

    return avgAcc, stdAcc, avgTrainAcc, str_acc, str_TrainAcc, str_sens, str_spec


def calculate_classification_detailes_from_cross_fold(metadata_with_classification_result):
    """
    Calculates different classification details: ROC_AUC, average distance to boarder, the best threshold and the values in the confusion matrix for each run. 
    
    Args: 
        metadata_with_classification_results (DataFrame): Dataframe with metadata and classification results. 
        
    Returns: Roc Auc value, Distance to boarder for correct classified, Distance to boarder for Wrongly classified, TN, TP, FN, FP, Best threshold for test set
    """
    true = np.array(metadata_with_classification_result['True class'])
    pred = np.array(metadata_with_classification_result['Predicted Class'])
    
    # ROC 
    roc_auc = round(roc_auc_score(true, np.array(metadata_with_classification_result["Distance to decition boarder"])), 2)
    
    # Distance to boarder: 
    correct_metadata = metadata_with_classification_result[metadata_with_classification_result['Correctly classified'] == True]
    wrong_metadata = metadata_with_classification_result[metadata_with_classification_result['Correctly classified'] == False] 
    
    # Average distance to boundery (Need abs value so that the samples on each side of the boarder does not balance each other out.)
    distance_to_boarder_correct = round(np.average(np.abs(correct_metadata['Distance to decition boarder'])), 2)
    distance_to_boarder_wrong = round(np.average(np.abs(wrong_metadata['Distance to decition boarder'])), 2)
    # distance_to_boarder_all = round(np.average(np.abs(metadata_with_classification_result['Distance to decition boarder'])), 2)
    
    # Conf mat values 
    confusion_mat = confusion_matrix(true, pred, labels=list(class_labels.values()))
    TN = confusion_mat[0, 0] # True Negative (True: HC, Pred: HC)
    TP = confusion_mat[1, 1] # True Positive (True: PD, Pred: PD)
    FN = confusion_mat[1, 0] # False Negative (True: PD, Pred: HC)
    FP = confusion_mat[0, 1] # False Positive (True: HC, Pred: PD)
    
    # Best threshold 
    fpr, tpr, thresholds = metrics.roc_curve(true, np.array(metadata_with_classification_result["Distance to decition boarder"]))
    best_threshold_idx = np.argmax(tpr - fpr)  # Choosen based on Youden's J statistic --> Maximising TPR - FPR
    best_threshold_test_set =  thresholds[best_threshold_idx]

    return roc_auc, distance_to_boarder_correct, distance_to_boarder_wrong, TN, TP, FN, FP, best_threshold_test_set






########## EXCELL FUNCTIONS ##################

def generate_timestamp():
    """
    Generates a string showing the date and time for the moment the code is executed. 
    
    Returns this string indicating a timestamp. 
    """
    local_timezone = pytz.timezone('Europe/Oslo') # Define the timezone
    local_time = datetime.now(local_timezone)
    timestamp = local_time.strftime("%d-%m-%Y_%H-%M-%S") # Format the filename
    return timestamp


def write_to_excel(data_to_excell, filename): 
    """
    Writes the information from data_to_excell to an exel file defined in filename.  
    
    Args: 
        data_to_excell (list): list of DataFrames that we want written to excell. 
        filename: Name of file we want the DataFrames written to. 
    """
    sheet_name = generate_timestamp()
    # Define an exel writer that creates a new file if file not already existing, creates a new sheet if sheet not already existing, writes over the existing sheet if sheet in file already existing. 
    exel_writer = pd.ExcelWriter(filename, mode="w") if not os.path.exists(filename) else pd.ExcelWriter(filename, mode="a", if_sheet_exists = "overlay")
    row = 1
    with exel_writer as writer:
        for i in range(len(data_to_excell)):
            data_to_excell[i].to_excel(writer, sheet_name=sheet_name, startrow=row , startcol=0, header=True)
            row += len(data_to_excell[i].index) + 3 # + 3 to get 2 lines between each new dataframe in the same sheet. 
            
    print(f"Data written to excell file {filename}, in sheet {sheet_name}.")
    print("---------------------------------------------------------------")


def create_feature_vs_acc_plot(results, image_path, train_acc = True):
    """ 
    Creates plot showing number of features vs accuracy. 
    
    Args: 
        results (DataFrame): DataFrame with results showing both accuracy and the number of features for each run. 
        image_path: Path to where we want the image saved. 
        train_acc (Bool): Indicates whether we want the training accuracy to be plotted in the graph as well (True) or only the test accuracy (False). 
    
    """
    acc_sorted_results = results.sort_values(by = ['Acc'], ascending = [False]) 

    x = np.array(results["Selected number of features"])
    y = np.array(results["Acc"])
    
    x_max = np.array(acc_sorted_results["Selected number of features"])[0]
    y_max = np.array(acc_sorted_results["Acc"])[0]
    
    if train_acc:
        y_train = np.array(results["TrainAcc"])
        plt.plot(x, y_train, label = "Train accuracy")
    
    plt.plot(x, y, label = "Test accuracy ")
    plt.scatter(x_max, y_max, color="red", label=f"Maximum accuracy is {y_max}% for {x_max} features", edgecolors='k', marker='o') # Plot of max -> Will be the first occurance of that value. 

    plt.xlabel('Number of features', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Evaluating number of features based on accuracy', fontsize=18)
    plt.legend(loc='lower right')
    plt.show()
    
    plt.savefig(image_path + ".png")
    
    plt.close()