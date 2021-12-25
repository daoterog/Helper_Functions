import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier

from scikit_feature_master.skfeature.function.similarity_based.SPEC import spec, feature_ranking

def variables_amount_deviation(df, variable_list, column_names):
    
    """
    Creates variables that states the deviation from the median scaled by the MAD
    of a chosen variable. x1 = (x0 - median(x0))/mad(x0)
    
    Args:
        df: DataFrame.
        variable_list: list of variable for which the deviation is calculated.
        column_names: list of names of the new variables ['sender_name', 'receiver_name']
        
    Return:
        df: with appender columns
    """
    
    # Define Variables that are going to be used
    global_variables = variable_list + ['amount']
    sender_variables = ['sender_type', 'account_from'] + global_variables
    receiver_variables = ['receiver_type', 'account_to'] + global_variables
    
    # Remove duplicates
    sender_variables = list(dict.fromkeys(sender_variables))
    receiver_variables = list(dict.fromkeys(receiver_variables))
    
    # Get median of the value and group by the desired variables
    # The value is calculated for the sender, receiver, and S2R type.
    aux_sender = df[sender_variables].groupby(sender_variables[:-1]).median().reset_index()
    aux_receiver = df[receiver_variables].groupby(receiver_variables[:-1]).median().reset_index()
    
    # Create a dictionary that maps the median of the value spended by month
    # by each of the identified groups
    aux_dict_sender = aux_sender.groupby(['sender_type'] + variable_list).median().to_dict()['amount']
    aux_dict_receiver = aux_receiver.groupby(['receiver_type'] + variable_list).median().to_dict()['amount']
    
    # Create Series objects that store the mapped dictionary values
    sender_median_ammount = df[['sender_type'] + variable_list].apply(lambda x: aux_dict_sender[tuple(x)], axis=1)
    receiver_median_ammount = df[['receiver_type'] + variable_list].apply(lambda x: aux_dict_receiver[tuple(x)], axis=1)
    
    #Substract the transaction amount to state the deviation
    sender_median_deviation_ammount = sender_median_ammount.sub(df.amount)
    receiver_median_deviation_ammount = receiver_median_ammount.sub(df.amount)
    
    # Auxiliar median deviation
    df['sender_abs_median_deviation'] = sender_median_deviation_ammount.abs()
    df['receiver_abs_median_deviation'] = receiver_median_deviation_ammount.abs()
    
    # Get the median of the absolute median deviation for each group
    aux_sender = df[sender_variables[:-1] + ['sender_abs_median_deviation']]\
                        .groupby(sender_variables[:-1]).median().reset_index()
    aux_receiver = df[receiver_variables[:-1] + ['receiver_abs_median_deviation']]\
                        .groupby(receiver_variables[:-1]).median().reset_index()
    
    # Create dictionary to map values the median absolute deviation of each group
    aux_dict_sender = aux_sender.groupby(['sender_type'] + variable_list).median()\
                            .to_dict()['sender_abs_median_deviation']
    aux_dict_receiver = aux_receiver.groupby(['receiver_type'] + variable_list).median()\
                            .to_dict()['receiver_abs_median_deviation']
    
    # Dfine Series objects that store mad values
    sender_mad_ammount = df[['sender_type'] + variable_list].apply(lambda x: aux_dict_sender[tuple(x)], axis=1)
    receiver_mad_ammount = df[['receiver_type'] + variable_list].apply(lambda x: aux_dict_receiver[tuple(x)], axis=1)
    
    # Add Columns to dataframe and drop the auxiliar ones created
    df[column_names[0]] = sender_median_deviation_ammount/sender_mad_ammount
    df[column_names[1]] = receiver_median_deviation_ammount/receiver_mad_ammount
    
    df.drop(columns=['receiver_abs_median_deviation', 'sender_abs_median_deviation'], inplace=True)
    
    return df

def perform_feauture_selection(X, y=None, mode='spec', n=None, n_sample=10000):

    """
    Performs feature selection according to the specified method (mode). spec uses
    Spectral Graph Theory to perform unsupervised feature selection. (recomended)

    Args:
        X (Dataframe): dataset.
        y (Series): dependant variable (binary encoded, fraud=1).
        mode (String): method to use.
        n (int): number of features to keep.
        n_sample (int): number of observations to include in 'sample-spec'.

    returns:
        df_feature_selection (Dataframe): dataset with columns filtered.
    """
    
    if mode == 'sample-spec':
        
        # Get sample
        X = X.sample(n=n_sample)
        
        # Extract the ranks of the features
        ranks = feature_ranking(spec(X.to_numpy()))

        # Get number of columns to get
        n_features = int(np.ceil(len(X.columns)*0.6))

        # Get columns indices, select them, and store them in a new df
        features = ranks[:n_features]
        keep_columns = X.columns[features]
        df_feature_selection = X[keep_columns]
                
    elif mode == 'spec':
        
        # Define K-Fold Split to make the processing quicker
        kf = KFold(n_splits=int(X.shape[0]/10000), shuffle=True)

        # List to store ranks assigned using SPEC methodology
        iteration_ranks = []

        # Loop to get ranks of each fold
        for _, indices in kf.split(X):

            X_spec = X.to_numpy()[indices, :]
            ranks = feature_ranking(spec(X_spec))
            iteration_ranks.append(ranks)

        # List to append mode of the previous iterations
        features = []
        feature_matrix = np.matrix(iteration_ranks)

        if n == None or n > len(X.columns):
            n = int(np.ceil(0.6*len(X.columns)))
        
        for i in range(n):
            
            mode = pd.Series(feature_matrix[:,i].tolist()).value_counts().sort_values(ascending=False)
            
            for value, _ in mode.iteritems():
                
                try:
                    _ = features.index(value[0])
                except ValueError:
                    features.append(value[0])
                    break
                else:
                    continue
           

        # Get columns to keep
        keep_columns = X.columns[features]
        df_feature_selection = X[keep_columns]
    
    elif mode == 'kbest':

        selector = SelectKBest()
        df_feature_selection = pd.DataFrame(selector.fit_transform(X, y))

    else: 
        
        model = ExtraTreesClassifier()
        model.fit(X,y)
        if n == None or n > len(X.columns):
            n = int(np.ceil(0.6*len(X.columns)))

        columns_to_keep = pd.Series(model.feature_importances_, index=X.columns).nlargest(n).index.tolist()

        df_feature_selection = X[columns_to_keep]

    return df_feature_selection