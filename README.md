# Project 1 — Cardiovascular Disease Risk Prediction
## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Usage](#usage)
- [Repo Structure](#repo-structure)
- [MilaLyon Team](#milalyon-team)
  
## Overview

This project aims to predict the likelihood of an individual developing Cardiovascular Diseases (CVDs), such as heart attacks or coronary heart disease (MICHD), using machine learning techniques. By analyzing lifestyle and health-related factors, the model classifies individuals into two categories:

- **-1**: Low risk
- **1**: High risk

The project follows the complete data science pipeline, from data preprocessing and feature engineering to model training, evaluation, and submission — and is part of the EPFL Machine Learning course.

## Dataset

The data originates from the Behavioral Risk Factor Surveillance System (BRFSS), which surveys U.S. residents about their health behaviors and chronic conditions.

The files provided are:

- **x_train.csv** — Training features (321 features, 328.135 samples)

- **y_train.csv** — Binary labels (+1 / -1) for training samples

- **x_test.csv** — Test features (109.379 samples)

Respondents were classified as having MICHD if a health professional had diagnosed them or they had experienced a heart attack or angina.


## Usage

To run the project, ensure you have a working Python environment with the necessary libraries installed. Only numpy and matplotlib are used.



The data can be extracted from `data/dataset.zip` and placed in the `data/` directory such that the structure looks like this:

```
data/
    x_train.csv
    y_train.csv
    x_test.csv
    features_description.csv
```

Run the following to predict using the selected model and hyper-parameters and to create the submission
> python run.py


### Optional: Conda Environment

For convenience and reproducibility, you can use the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate milalyon
```


## Repo Structure

The repository is organized as follows:

- `run.py` — Main training and submission script for the CVD Prediction project.

- `data/` — Contains the dataset files.
    - x_train.csv
    - y_train.csv
    - x_test.csv
    - features_description.csv # All the features with encoded information, such as relevant missing values, feature types...

- `src/` — Contains the different functions.
    - `helpers.py` - Utility functions for data loading and submission creation. It loads data from CSV files and the features_description.csv file necessarry for preprocessing.
    - `preprocessing.py` — Functions for data cleaning and preprocessing. The function preprocess_data gathers all preprocessing steps and returns the cleaned data.
    - `implementations.py` — Contains implementations of machine learning algorithms used in the project.
    - `model_evaluation.py` — Evaluation of model performance, including accuracy and F1-score calculations.
    - `crossvalidation.py` — Functions for tuning hyperparameters using cross-validation. cross_validate_hyperparameter function performs k-fold cross-validation and can be used for any model and hyperparameter.

- `notebooks/` — Contains the Jupyer Notebook files.
    - `crossvalidation.ipynb` — Jupyter Notebook for hyperparameter tuning using cross-validation.
    - `run_no_process.ipynb` — Jupyter Notebbok for training logistic regression without applying full preprocessing.
    - `run_no_process_poly.ipynb` — Jupyter Notebbok for training logistic regression with polynomial features without applying full preprocessing.
    - `run.py` — Main script to execute the entire pipeline from data loading to model training and submission generation. The hyperparameters can be adjusted in this file but are set to optimal values found during cross-validation by default.

- `report/` — Contains the Report and the Project Description.
    - `project1_description.pdf`
    - `Report.pdf`


## MilaLyon Team

- Beatrice Campo
- Nathan Mutin
- Valentin Planes

EPFL Machine Learning — Project 1 (Fall 2025)
