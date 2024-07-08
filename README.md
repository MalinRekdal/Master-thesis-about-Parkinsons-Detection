# Master-thesis for Malin Rekdal
 - Title: Speech Analysis for Automatic Parkinson's Disease Detection
 - Sub title: Feature, data, and language analysis for performance optimization.

Supervisor: Torbjørn Karl Svendsen

 This code is created for a master thesis in Electronic System Design and Innovation at NTNU. 

## 1 -  Main files to use in this repository: 
- #### Code: 
  - Setup
    - readMe: this file with an introduction to the code. 
    - constants: constant definitions that is used throughout the other code
    - functions: function definitions that is used throughout the other code
    - project_setup: Setup file to run things that is common for many files in the beginning. 
    - create_features_from_wav: python script that creates a excell document with the feature values for the speech recordings in PC-GITA and EWA-DB. 
    - requirements.txt: Technical requirements for this project to work. 

  - Feature and data definitions 
    - features_to_choose_from: md file with an overview of the Phonation, Articulation and Prosody features we can choose between. 
    - definition_of_data_and_feature_groups: Definition of the different feature and data sub groups used throughout the project. 
    - wav_paths: All paths to the waveforms for PC-GITA and EWA-DB that will be used to extract features. 

  - Classification 
    - svm_model: Script that can be runned with different data and feature combinations and predicts PD vs HC based on a SVM model.  
    - svm_model_diff_datasets: Used to evaluate how the model works when we train on one database and test on another. 
    - svm_model_combined_dataset: Used to evaluate how the model works when we train and test on a combination of the different databases. 
    - GITA_model_detailed_results: Predicts PD vs HC based on a feature - data combination, and gives a more detailed analysis of the results for PC-GITA. 
    - grid_search: File to do grid search on SVM with different data and features to find the optimal parameter values.




  - Detailed analysis of data and features: 
    - analyse_metadata files: jupyter notebook that analyses the metadata from PC-GITA and EWA-DB.
    - analyse_signal files: jupyter notebook that analyses the waveform signals from PC-GITA, EWA-DB and wav_paths and extracts some features 
    - feature_evaluation: Notebook that plots different features both alone and together with other features. 
    - results_plots: Plots some different results where the values are obtained trough grid search or svm_model, but plotted later on. 
    - GITA_zero_and_NaN analysis: Investigates zero and NaN problems with the features extracted from DisVoice. 
    - best_features: Sequential Feature Selection and Random method to find ideal feature subsets. 

  - #### Balanced_folds: Code to create fold distributions and files showing the created distributions for the PC-GITA and EWA-DB databases. 
- #### Generated_results: Plots or images generated through other code will be saved here. 
- #### Images: Folder with images showing results that are used in the report for the project. 
- #### Excell Features: Features extracted from EWA-DB and PC-GITA.  
- #### Excell Results: All results obtained for this project used in the report. 


## 2 - Getting started


### 2 choises for what to do: 
1) #### It is only the create_features_from_wav and analyse_signal files that needs the setup with DisVoice, Praat and Kaldi. Therefore you can choose to skip the steps underneath and just use the features in Features.xlsx file that is attached and run the rest of the code as normal. Then the packages you need must be installed separately as well. 

2) #### If you do the steps underneath and set up then you can run create_features_from_wav to create features for all data. This file needs to be run with one system argument that is a string, and corresponds to the name of a file you want the features saved to. A file will then be created with the phonation, articulation and prosody features from all waveforms indicated with the paths. Then you use this file of features from the PC-GITA-data file and run the rest of the code.

### The steps underneath will explain how to set up an environment on a Linux device and install everything needed for this project. This is based on my experiences and the problems I acountered.  


### 2.1 - Setup environment

- Install conda or Miniconda
  - Installation of miniconda: 
    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
    bash Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc
    rm -rf Miniconda3-latest-Linux-x86_64.sh
    ```


- Set up virtual env 
  - Create environment 

    ```
    conda create -n disenv python=3.11.5 ipython ipykernel
    ```

  - Activate and deactivate environment
    ```
    conda activate disenv
    ```
    ```
    conda deactivate
    ```
- Create a kernal using: 
  ```
  python -m ipykernel install --user --name disenv --display-name "disenv"
  ```



### 2.2 - Make sure pip is linked to the virtual environmentet 
- Use
  ```
  which pip
  ```
- If this is not the pip related to the env you have created you need to add an alias to the .bashrc file by adding this line at the end: 
  ```
  alias disenvPipp=/path_to_miniconda/miniconda3/envs/disenv/bin/pip
  ```
  Then use disenvPipp instead of pip when installing things. 

### 2.3 - Activate conda and environment when terminal starts.  
- To automatically activate conda and the environement when the terminal start you can create a .bash_profile file. 
  - If a .bash_profile file don't exist, create one. 
  - To automatically start conda add this line to the bash_profile file: 
    ```
    source~/.bashrc
    ```
  - To automatically activate environment add this line to the bash_profile file: 
    ```
    conda activate disenv
    ```

- Note: If conda starts on its own on default and you dont care if you need to write conda activate disenv every time a new terminal is activated then 2.3 can be skipped. 


### 2.4 - Install requirements to conda environment 
- Install praat: 
  ```
		wget https://www.fon.hum.uva.nl/praat/praat6318_linux64nogui.tar.gz 
		gunzip praat6318_linux64nogui.tar.gz
		tar xvf praat6318_linux64nogui.tar
		rm praat6318_linux64nogui.tar
		mv praat_nogui praat 
  ```

  - Add path to praat by adding a this line to .bashrc: 
    ```
      export PATH=.:${HOME}/master-thesis:$PATH
    ```
  - Note: For this experiment version 6.3.18 is used, and works well. The experiments are also tested with 6.4.05 and 6.4.06 but those seem to give some problem when extracting articulation features. 

- Install requirements from requirements.txt file: 
    ```
    pip install -r requirements.txt
    ```
  Or manually install the requirements. 
  - Install DisVoice to same location as Praat

    ```
    pip install disvoice
    ```
    Note that pip install disvoice installs both pythorch and tensorflow and the other packages that is needed to use DisVoice. 

      - Can also clone Disvoice repo if nootebook examples are wanted: 
        ```
        git clone https://github.com/jcvasquezc/DisVoice.git
        ```
  - Other requirements needed:
    ```
    pip install openpyxl
    pip install scipy==1.10.0
    pip install scikit-learn 1.5.0 [Important that it is 1.5.0]
    ```

- Install kaldi: 
  ```
		git clone https://github.com/kaldi-asr/kaldi
  ```
  Follow the instructions in the INSTALL file in order to install the toolkit. This might need an earlier version of python, and in that case you can create such an environment like this: 
  ```
		conda create -name disvoice27 python=2.7
  ```
  
  Do the installation and then go back to the disenv environment created earlier with a newer python version. 


### 2.5 - Changes needed to DisVoice package: 
  - In miniconda/env/disenv you need to change 3 files: 

    - ./miniconda3/envs/disvoice/lib/python3.11/sitepackages/disvoice/articulation/articulation_functions.py
        - line 68: change np.int to np.int64

    - ./miniconda3/envs/disvoice/lib/python3.11/site-packages/disvoice/praat/praat_functions.py
      -	change the beginning of lines 44 and 68
        -	from command='praat '+PATH… 
        -	to command='praat --run '+PATH... 
        - Note: important to have " " after "--run". 

    - /miniconda3/envs/disenv/lib/python3.11/site-packages/disvoice/phonation/phonation_functions.py"
      - change line 18 and 57: from sys.warn to warnings.warn. 
      - Change line 11 from import sys to import warnings. 
      - This is due to the fact that sys does not have an attribute called warning. 



## 3 - Update constant.py according to your setup: 
- personal_path_to_disvoice = the path to where DisVoice repo is cloned to. 
- personal_path_to_kaldi = the path to where kaldi repo is cloned to.
- personal_path_to_code =  the path to where this repo is cloned to.
- personal_path_to_PC_GITA and personal_path_to_EWA = path to the PC-GITA and EWA-DB databases. 
  - The databases are available on deepthought.ies.ntnu.no (computational server solution "marwin" available through NTNU) in talebase/data/speech_raw. Using this data can be done in one of three methods [Explained for PC-GITA, but is similar for EWA-DB] : 
    1) can be a link to the data that is located in: /talebase/data/speech_raw
    2) Can copy the data to a location of your choise in Aulus like this: 
        ```
        mkdir PC-GITA
        ln -s /talebase/data/speech_raw/PC-GITA/* path/to/new/location 
        ```
    3) Can copy the data to a local computer (windows example below) to work on it offline: 
        - Do step 2. 
        - Download putty and add putty to path. 
        - Copy the data using putty: 
          ```
          pscp -r username@deepthought.ies.ntnu.no:\path/to/new/location/for/PC-GITA  C:\Users\username\path\to\local\location
          ```
  
