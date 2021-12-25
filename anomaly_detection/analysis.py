import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prince import PCA
from sklearn.model_selection import train_test_split

from frameworks import (evaluate_metalearner, unsupervised_representation_learning,
                        accuracy_tos)

def make_prediction(metalearner, X_tuple, y_tuple, threshold, accuracy,  
                    representation_learning_model_dict, mode, n_layers):
    
    """
    Makes predictions using the specified methodology.
    
    Args:
        model (Object): anomaly detector or classifier.
        X_tuple: tuple containing (X_train, X_test).
        y_tuple: tuple containing (y_train, y_test).
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        
    Returns:
        pred_tuple: tuple of Dataframes containing predictions for each dataset form -> (training_set_df, testing_set_df).
        X_dict (Dictionary): dictionary with used_dictionary.
    
    """
    
    # Unpack tuple parameters
    (X_train, X_test) = X_tuple
    (y_train, y_test) = y_tuple
    
    # Define prediction tuple (statement that is going to be returned)
    pred_tuple = 0
    
    # Define auxiliar dictionary in order to use the evaluate_metalearner function
    X_dict = {}
    
    # Choose prediction mode
    if mode == 'regular':
        
        # Build dataset dictionary (necessary in cross_valiation_framework as well)
        X_dict = {mode: (X_train, X_test)}
        
        # Evaluate learner and make predictions
        (_, _, _, _, pred_tuple, raw_decision_tuple) = evaluate_metalearner(metalearner, X_dict, y_train, y_test, threshold)
        
    else:
        
        # Check if representation learner are defined
        if representation_learning_model_dict != None:
            
            # Create representation learning dataset
            (df_decision_scores, df_decision_function, 
                df_training_prediction, df_testing_prediction) = unsupervised_representation_learning(representation_learning_model_dict, 
                                                                                                    X_train, 
                                                                                                    X_test, 
                                                                                                    threshold)
            
            # Perform Feature Selection
            features_to_keep = accuracy_tos(df_decision_scores, df_decision_function, df_training_prediction, 
                                            df_testing_prediction, y_train, y_test, accuracy)
            
            # Filter Learners
            df_filtered_decision_scores = df_decision_scores[features_to_keep]
            df_filtered_decision_function = df_decision_function[features_to_keep]
            df_filtered_training_prediction = df_training_prediction[features_to_keep]
            df_filtered_testing_prediction = df_testing_prediction[features_to_keep]
    
            # Build each dataset
            X_comb_training = X_train.join(df_decision_scores, how='outer')
            X_comb_testing = X_test.join(df_decision_function, how='outer')
            
            # Enable additionaal layers of stacking
            for j in range(n_layers):

                print(f'{j+1}th layer of stacking')

                # Create representation learning dataset using the combined dataset (because is the second layer)
                (df_decision_scores_n, df_decision_function_n, 
                df_training_prediction_n, df_testing_prediction_n) = unsupervised_representation_learning(representation_learning_model_dict, 
                                                                                                    X_comb_training, 
                                                                                                    X_comb_testing, 
                                                                                                    threshold)
                
                # Rename columns to avoid overlapping
                df_decision_scores_n.columns = [learner + '-' + str(j) for learner in df_decision_scores_n.columns]
                df_decision_function_n.columns = [learner + '-' + str(j) for learner in df_decision_function_n.columns]
                df_training_prediction_n.columns = [learner + '-' + str(j) for learner in df_training_prediction_n.columns]
                df_testing_prediction_n.columns = [learner + '-' + str(j) for learner in df_testing_prediction_n.columns]

                # Join Decision Scores
                df_filtered_decision_scores = df_filtered_decision_scores.join(df_decision_scores_n, how='outer')
                df_filtered_decision_function = df_filtered_decision_function.join(df_decision_function_n, how='outer')
                df_filtered_training_prediction = df_filtered_training_prediction.join(df_training_prediction_n, how='outer')
                df_filtered_testing_prediction = df_filtered_testing_prediction.join(df_testing_prediction_n, how='outer')

                # Perform Feature Selection
                features_to_keep_n = accuracy_tos(df_filtered_decision_scores, df_filtered_decision_function, df_filtered_training_prediction, 
                                                df_filtered_testing_prediction, y_train, y_test, accuracy)

                # Filter Learners
                df_filtered_decision_scores = df_filtered_decision_scores[features_to_keep_n]
                df_filtered_decision_function = df_filtered_decision_function[features_to_keep_n]
                df_filtered_training_prediction = df_filtered_training_prediction[features_to_keep_n]
                df_filtered_testing_prediction = df_filtered_testing_prediction[features_to_keep_n]
                
                # Build each dataset
                X_comb_training = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing = X_test.join(df_filtered_decision_function, how='outer')
            
            # Fix datasets depending of number of stacking layers
            if n_layers == 0:
                X_comb_training_filtered = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing_filtered = X_test.join(df_filtered_decision_function, how='outer')
            else:
                X_comb_training = X_train.join(df_decision_scores, how='outer')
                X_comb_testing = X_test.join(df_decision_function, how='outer')
                X_comb_training_filtered = X_train.join(df_filtered_decision_scores, how='outer')
                X_comb_testing_filtered = X_test.join(df_filtered_decision_function, how='outer')
            
            # Build Dataset Dictionary
            if mode == 'stacking-decisions': 
                X_dict = {mode: (df_decision_scores, df_decision_function)}
            elif mode == 'stacking-decisions-filtered':
                X_dict = {mode: (df_filtered_decision_scores, df_filtered_decision_function)}
            elif mode == 'stacking-decisions-combined':
                X_dict = {mode: (X_comb_training, X_comb_testing)}
            else:
                X_dict = {mode: (X_comb_training_filtered, X_comb_testing_filtered)}
                
            # Evaluate Meta-learner over each dataset
            (_, _, _, _, pred_tuple, raw_decision_tuple) = evaluate_metalearner(metalearner, X_dict, y_train, y_test, threshold)
            
        else:
            print('You need to define a dictionary of representation learners')
            pred_tuple = None
            
    return pred_tuple, X_dict


def transcation_report(metalearner, X, y, month_indicator, channel_indicator, type_indicator, test_size, 
                       threshold=0.99, accuracy='precision', representation_learning_model_dict=None, 
                       mode='stacking-filtered', n_layers=0, path=None):
    
    """
    Generates reports regarding the quantity of transactions blocked by type and by channel and 
    compares it with the real amount. Plots the percentage of transactions that are blocked per month.
    
    Args:
        model (Object): anomaly detector or classifier.
        X (Dataframe): dataset (independet variables).
        y (Series): independent variables.
        k (int): number of folds for cross validatioon.
        metalearner_name (String): name of learner.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        rep_learner_codes (String): representation learner codes,
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        path (String): path to save figures and results.
        
    Return:
        df_channel_fraud (Dataframe): Report of number of blocked transactions by channel.
        df_type_fraud (Dataframe): Report of number of blocked transactions by type.
    """
    
    # Split the dataset into training and testing sets
    cut = int(np.ceil(X.shape[0]*(1 - test_size)))
    X_train = X.iloc[:cut, :]; X_test = X.iloc[cut:, :]
    y_train = y.iloc[:cut]; y_test = y.iloc[cut:]
    month_train = month_indicator.iloc[:cut]; month_test = month_indicator.iloc[cut:]
    
    # Make prediction with classifier
    pred_tuple, _ = make_prediction(metalearner, (X_train, X_test), (y_train, y_test), threshold, accuracy, 
                        representation_learning_model_dict, mode, n_layers)
    
    # Unpack prediction tuple
    (y_pred_train, y_pred_test) = pred_tuple
    
    # Append month indicator to predictions, group them by month, and get percentage of how many
    # transactions are blocked by month
    y_pred_train['month'] = month_train
    y_pred_test['month'] = month_test
    y_pred_total = y_pred_train.append(y_pred_test)
    total_transactions_per_month = y_pred_total.groupby('month').count()[mode]
    blocked_transactions_per_month = y_pred_total.groupby('month').sum()[mode]
    percentage_blocked_transactions_per_month = blocked_transactions_per_month/total_transactions_per_month
    
    # Use same methodology to get the real percentage of fraudulent montly transactions
    y = y.to_frame()
    y['month'] = month_indicator
    fraud_transactions_per_month = y.groupby('month').sum()['fraud']
    percentage_fraud_transactions_per_month = fraud_transactions_per_month/total_transactions_per_month
    
    intersection_month = month_indicator.iloc[cut]
    
    # Get how many frauds where identified by channel and by transaction_mode and contrast it
    # with the real statistics
    y_pred_total['channel'] = channel_indicator; y['channel'] = channel_indicator
    y_pred_total['transaction_mode'] = type_indicator; y['transaction_mode'] = type_indicator 
    y_pred_channel_blocked = y_pred_total[['channel', mode]].groupby('channel').sum()[mode]
    y_channel_fraud = y[['channel', 'fraud']].groupby('channel').sum()['fraud']
    y_pred_type_blocked = y_pred_total[['transaction_mode', mode]].groupby('transaction_mode').sum()[mode]
    y_type_fraud = y[['transaction_mode', 'fraud']].groupby('transaction_mode').sum()['fraud']
    y_percentage_channel_blocked = y_pred_channel_blocked/y_channel_fraud
    y_percentage_type_blocked = y_pred_type_blocked/y_type_fraud
    df_channel_fraud = pd.DataFrame({'Predicted':y_pred_channel_blocked, 'Real': y_channel_fraud,
                                     'Rate': y_percentage_channel_blocked})
    df_type_fraud = pd.DataFrame({'Predicted':y_pred_type_blocked, 'Real': y_type_fraud,
                                     'Rate': y_percentage_type_blocked})
    
    # Plot
    t = list(range(1,13))
    
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, percentage_blocked_transactions_per_month, color='navy', label='Precentage of Blocked Transactions')
    ax.scatter(t, percentage_blocked_transactions_per_month, marker='o', color='red')
    ax.plot(t, percentage_fraud_transactions_per_month, color='red', label='Precentage of Fraudulent Transactions')
    ax.scatter(t, percentage_fraud_transactions_per_month, marker='o', color='red')
    ax.axvline(intersection_month, color='lime')
    ax.legend(loc='best')
    ax.set_xlabel('month')
    ax.set_title('Model Blocked transactions per month')
    
    if path:
        df_channel_fraud.to_csv(os.path.join(path,'channel_report.csv'))
        df_type_fraud.to_csv(os.path.join(path,'transaction_mode_report.csv'))
        fig.savefig(os.path.join(path,'monthly_blocked_transactions.png'), dpi=100)
    
    return df_channel_fraud, df_type_fraud

def plotting_outlier(metalearner, X, y, test_size, threshold=0.99, accuracy='precision', 
                    representation_learning_model_dict=None, mode='stacking-filtered', n_layers=0, path=None):
    
    """
    Plots the dataset by reducing its dimensionality to 2 components using PCA. Tool to visualize outlier.
    
    Args:
        model (Object): anomaly detector or classifier.
        X (Dataframe): dataset (independet variables).
        y (Series): independent variables.
        k (int): number of folds for cross validatioon.
        metalearner_name (String): name of learner.
        threshold (float): threshold that states where to regard an observation as an anomaly (in [0,1]).
        accuracy (String): accuracy metric to evaluate learners.
        representation_learning_model_dict (Dictionary): representation learners dicionary.
        rep_learner_codes (String): representation learner codes,
        mode (String): cross validation framework.
        n_layers (int): number of additional layers of stacking.
        path (String): path to save figures and results.

    Output:
        2D dataset plot differenced by label
    """

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Make prediction with classifier
    pred_tuple, X_dict = make_prediction(metalearner, (X_train, X_test), (y_train, y_test), threshold, accuracy, 
                        representation_learning_model_dict, mode, n_layers)
            
    # Initalize PCA object and get coordinates
    pca_prince = PCA(n_components=2, n_iter=3, rescale_with_mean=False, rescale_with_std=False)
    coordinates_train = pca_prince.fit_transform(X_dict[mode][0]).astype(float)
    coordinates_train['pred'] = pred_tuple[0].astype('category')
    coordinates_train.columns = ['cord1', 'cord2', 'pred']
    coordinates_test = pca_prince.fit_transform(X_dict[mode][1]).astype(float)
    coordinates_test['pred'] = pred_tuple[1].astype('category')
    coordinates_test.columns = ['cord1', 'cord2', 'pred']
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(data=coordinates_train, x='cord1', y='cord2', hue='pred', ax=ax[0])
    ax[0].set_xlabel('First Component')
    ax[0].set_ylabel('Second Component')
    ax[0].set_title('Training 2 Component PCA Projection')
    sns.scatterplot(data=coordinates_test, x='cord1', y='cord2', hue='pred', ax=ax[1])
    ax[1].set_xlabel('First Component')
    ax[1].set_ylabel('Second Component')
    ax[1].set_title('Testing 2 Component PCA Projection')

    if path:
        title = 'oulier_pca_2dim_plot.png'
        fig.savefig(os.path.join(path, title), dpi=100)