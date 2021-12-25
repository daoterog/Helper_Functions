import os, shutil, errno, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            average_precision_score, roc_curve, auc, plot_precision_recall_curve)

from evaluation import (find_most_wrong_prediction, across_class_results, 
                        make_confusion_matrix, precision_recall_barplot)

def multiple_model_hyperparameter_tuning(model_list, hyperparameter_list, 
                                         X_dict, y, k, scoring):
    
    """
    Get hyperparameter dictionary and store it in a list.
    
    Args:
        model_list: classifier list.
        hyperparameter_list: hyperparameter grid list.
        X_dict: dataset dictionary.
        y: dependent variable.
        k: number of folds for cross-validation.
        scoring: reference metric to evaluate tuning.
    """
    
    # Store results
    param_dict_list = []
    
    for i, classifier in enumerate(model_list):
        
        # Get hyperparameter grid
        param_grid = hyperparameter_list[i]
        
        # Get hyperparameter dictionary
        param_dict, _ = hyperparametertunning(classifier, X_dict, y, param_grid, 
                                              k, scoring)
        
        # Store result
        param_dict_list.append(param_dict)
        
    return param_dict_list

def get_multiple_model_predictions(model_list, k, X_dict, y, param_dict_list, image_filenames):

    """
    Generates predictions for each event for each model and append it to dataset.

    Args:
        classifier: model.
        k: number of splits.
        X_dict: dataset dictionary.
        y: labels.
        param_dict: parameter dictionary for the model.
        param_title_dictionary: model hyperparameter title (String).
        image_filenames: array with image filenames
        
    Output:
        new_X_dict: dictionary of new datasets.
    """

    new_X_dict = {}

    for dataset_key in X_dict.keys():

        X = X_dict[dataset_key].to_numpy()

        # Set parameters
        tuned_model_list = [classifier.set_params(**param_dict_list[i][dataset_key])\
                            for i, classifier in enumerate(model_list)]

        skf = StratifiedKFold(n_splits=k, shuffle=True)

        # Store Results
        model_probabilities = np.empty(0)
        test_indexes = []

        for train_index, test_index in skf.split(X, y):

            # Split data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Store Results
            predicted_probabilities = np.empty(0)

            # Loop for tuned models
            for tuned_model in tuned_model_list:

                # Fit the model and predict
                tuned_model.fit(X_train, y_train)
                pred_probs = tuned_model.predict_proba(X_test)

                # Store probabilities
                if predicted_probabilities.size == 0:
                    predicted_probabilities = pred_probs
                else:
                    predicted_probabilities = np.hstack((predicted_probabilities, pred_probs))

            # Store indexes and stack probabilities
            test_indexes.extend(test_index)
            if model_probabilities.size == 0:
                model_probabilities = predicted_probabilities
            else:
                model_probabilities = np.vstack((model_probabilities, predicted_probabilities))

        # Get ordered filenames
        test_filenames = image_filenames[test_indexes]

        # Store it in a DataFrame
        filename_probability = pd.DataFrame(model_probabilities)
        filename_probability.index = test_filenames

        # Order them by filename
        filename_probability = filename_probability.loc[image_filenames, :]

        # Add probabilities to dataset and add it to dictionary
        X_probability = np.hstack((X, filename_probability))
        new_X_dict[dataset_key] = pd.DataFrame(X_probability)

    return new_X_dict

def get_sub_model_predictions(classifier, k, X_dict, y, param_dict, image_filenames):
    
    """
    Generates predictions for each event and append it to dataset.
    
    Args:
        classifier: model.
        k: number of splits.
        X_dict: dataset dictionary.
        y: labels.
        param_dict: parameter dictionary for the model.
        param_title_dictionary: model hyperparameter title (String).
        image_filenames: array with image filenames
        
    Output:
        new_X_dict: dictionary of new datasets.
    """
    
    new_X_dict = {}
    all_probs = np.empty(0)
    
    for dataset_key in X_dict.keys():

        X = X_dict[dataset_key].to_numpy()

        # Set parameters
        classifier.set_params(**param_dict[dataset_key])

        skf = StratifiedKFold(n_splits=k, shuffle=True)
        
        # Store Results
        predicted_probabilities = np.empty(0)
        test_indexes = []
        
        for train_index, test_index in skf.split(X, y):

            # Split data
            X_train, X_test = X[train_index], X[test_index]
            y_train, _ = y[train_index], y[test_index]

            # Fit the model and predict
            classifier.fit(X_train, y_train)
            pred_probs = classifier.predict_proba(X_test)
            
            # Store indexes and predicted probabilities
            test_indexes.extend(test_index)
            if predicted_probabilities.size == 0:
                predicted_probabilities = pred_probs
            else:
                predicted_probabilities = np.vstack((predicted_probabilities, pred_probs))
        
        # Get ordered filenames
        test_filenames = image_filenames[test_indexes]
        
        # Store it in a DataFrame
        filename_probability = pd.DataFrame(predicted_probabilities)
        filename_probability.index = test_filenames
        
        # Order them by filename
        filename_probability = filename_probability.loc[image_filenames, :]
        
        # Add probabilities to dataframe to then create dataset with all predictions
        if all_probs.size == 0:
            all_probs = filename_probability.to_numpy()
        else:
            all_probs = np.hstack((all_probs, filename_probability))
        
        # Add probabilities to dataset and add it to dictionary
        X_probability = np.hstack((X, filename_probability))
        new_X_dict[dataset_key + '_single_probability'] = pd.DataFrame(X_probability)
        
    # Loop to add dataset with all predictions
    for dataset_key in X_dict.keys():
        
        X = X_dict[dataset_key].to_numpy()
        
        # Add probabilities to dataset and add it to dictionary
        X_probability = np.hstack((X, all_probs))
        new_X_dict[dataset_key + '_multiple_probabilities'] = pd.DataFrame(X_probability)
        
    return new_X_dict

def hyperparametertunning(classifier, X_dict, y, param_grid, k, scoring):

    """
    Tunes the hyperparameters of a given classifier using grid search.

    Args:
        classifier: model used to perform predictions.
        X_dict: dataset dictionary.
        y: dependent variable.
        param_grid: hyperparameter grid.
        k: number of folds for cross-validation.
        scoring: reference metric to evaluate tuning.

    Outputs:
        param_dict: models optimal hyperparameter grid.
        param_title_dictionary:
    """

    # Best Parameter dictionary
    param_dict = {}

    # Parameter title dictionary
    param_title_dictionary = {}

    # Specify CV
    skf = StratifiedKFold(n_splits = k)

    # Test each dataset
    for dataset_key in X_dict:

        X_train = X_dict[dataset_key]

        clf = GridSearchCV(classifier, param_grid, cv  = skf, scoring = scoring, n_jobs=-1)
        clf.fit(X_train, y)

        # Store parameters
        param_dict[dataset_key] = clf.best_estimator_.get_params()

        hyperparameter_title = ''
        for hyperparameter in param_grid.keys():
            hyperparameter_title += hyperparameter + '=' + \
                        str(param_dict[dataset_key][hyperparameter]) + '-'

        # Store titles
        param_title_dictionary[dataset_key] = hyperparameter_title[:-1]

    return param_dict, param_title_dictionary

def learningcurve(classifier, X_dict, y, cv, param_dict, scoring, train_sizes):
    """ Calculate the learning curve values.

    Args:
        classifier: model used to perform the prediction
        X_dict: datasets dictionary
        y: labels
        cv: number of cross-validation splits
        param_dict: parameter dictionary
        scoring: metric used to evaluate cross validation
        train_sizes: specified train sizes

    Output:
        train_sizes_dict: train_sizes for each dataset considered (dictionary).
        train_scores_mean_dict: mean of the training scores for each train size 
            (dictionary).
        train_scores_std_dict: self explanatory.
        test_scores_mean_dict: self explanatory.
        test_scores_std_dict: self explanatory.
    """

    train_sizes_dict = {}
    train_scores_mean_dict = {}
    train_scores_std_dict = {}
    test_scores_mean_dict = {}
    test_scores_std_dict = {}

    for dataset_key in X_dict:

        X_train = X_dict[dataset_key]

        # Set Parameters
        classifier.set_params(**param_dict[dataset_key])

        train_sizes, train_scores, test_scores = learning_curve(classifier, 
                                    X_train, y, 
                                    cv = cv, 
                                    scoring = scoring, 
                                    train_sizes = train_sizes)

        train_scores_mean = np.mean(train_scores, axis = 1)
        train_scores_std = np.std(train_scores, axis = 1)
        test_scores_mean = np.mean(test_scores, axis = 1)
        test_scores_std = np.std(test_scores, axis = 1)

        train_sizes_dict[dataset_key] = train_sizes
        train_scores_mean_dict[dataset_key] = train_scores_mean
        train_scores_std_dict[dataset_key] = train_scores_std
        test_scores_mean_dict[dataset_key] = test_scores_mean
        test_scores_std_dict[dataset_key] = test_scores_std

    return (train_sizes_dict, train_scores_mean_dict, train_scores_std_dict,
            test_scores_mean_dict, test_scores_std_dict)

def plotlearningcurve(model_name, param_dict, param_title_dictionary, score, 
                      train_sizes_dict, train_scores_mean_dict, 
                      train_scores_std_dict, test_scores_mean_dict,
                      test_scores_std_dict, path):
  
    """ 
    To plot the learning curve.

    Args:
        model_name: model name (String).
        param_dict: model parameter grid (dictionary).
        param_title_dictionary: model hyperparameter title (String).
        score: ylabel. Metric with which the model is evaluated (String).
        train_sizes_dict: train_sizes for each dataset considered (dictionary).
        train_scores_mean_dict: mean of the training scores for each train size 
            (dictionary).
        train_scores_std_dict: self explanatory.
        test_scores_mean_dict: self explanatory.
        test_scores_std_dict: self explanatory.
        path: path to load the graphs.

    """
    
    # Loop for each dataset
    for dataset_key in train_sizes_dict:

        train_sizes = train_sizes_dict[dataset_key]
        train_scores_mean = train_scores_mean_dict[dataset_key]
        train_scores_std = train_scores_std_dict[dataset_key]
        test_scores_mean = test_scores_mean_dict[dataset_key]
        test_scores_std = test_scores_std_dict[dataset_key]

        # Experiment Title
        title = model_name + '_' + dataset_key + '_' + \
                                            param_title_dictionary[dataset_key]

        plt.figure()
        plt.title(title)
        plt.xlabel('Training examples')
        plt.ylabel(score)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, 
                            color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        plt.legend(loc="best")

        # Save Figure
        filename = 'learningcurve_'+ str(title) + '.jpg'
        plt.savefig(os.path.join(path,filename), dpi = 100)
        
        plt.show()

def multiclass_CV(classifier, k, X_dict, y, param_dict, param_title_dictionary, 
                  class_names, model_name, path, image_filenames):
    
    """
    Plots a confusion matrix and f1-scores calculated from the concatenation of 
    the results of the classifier over the whole dataset.
    Args:
        classifier: model.
        k: number of splits.
        X_dict: dataset dictionary.
        y: labels.
        param_dict: parameter dictionary for the model.
        param_title_dictionary: model hyperparameter title (String).
        model_name: self explanatory.
        path: to load the data.
    Output:
        df: DataFrame with the results of each model.
    """

    model_wrong_preds = {}

    df = pd.DataFrame()
    
    for dataset_key in X_dict.keys():

        X = X_dict[dataset_key].to_numpy()

        # Set parameters
        classifier.set_params(**param_dict[dataset_key])

        # Experiment Title
        title = model_name + '_' + dataset_key + '_' + \
                                            param_title_dictionary[dataset_key]

        skf = StratifiedKFold(n_splits=k)

        pred_labels = []
        true_labels = []
        test_indexes = []
        prob_labels = []

        for train_index, test_index in skf.split(X, y):

            # Split data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model and predict
            classifier.fit(X_train, y_train)
            pred_probs = classifier.predict_proba(X_test)
            y_pred = pred_probs.argmax(axis=1)
            y_prob = pred_probs.max(axis=1)

            # Make confusion matrix
            pred_labels.extend(y_pred)
            true_labels.extend(y_test)
            test_indexes.extend(test_index)
            prob_labels.extend(y_prob)

        # Find most wrong predictions
        model_wrong_preds[dataset_key] = find_most_wrong_prediction(image_filenames, 
                                            test_indexes, true_labels, pred_labels, 
                                            prob_labels, class_names, title, path)

        fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

        class_scores, accuracy, df_results = across_class_results(true_labels, 
                                                    pred_labels, class_names,
                                                    title, fig, ax[1])
        make_confusion_matrix(true_labels, pred_labels, fig, ax[0], accuracy,
                               class_names, norm=True)
        fig.suptitle(title)

        # Save Figure
        filename = 'cv_' + str(title) + '.jpg'
        fig.savefig(os.path.join(path,filename), dpi = 100)
        plt.show()

        # Make Second Plot
        precision_recall_barplot(class_scores, title, path)

        # Create and Store DataFrame
        df = pd.concat([df, df_results], ignore_index=True, axis=0)
        df.to_csv(os.path.join(path,os.path.basename(path) + '.csv'))

    return df, model_wrong_preds

def create_wrong_prediction_image_dir(image_paths, dir):
  
  """
  Creates a provisional directory to store images wrongfully classified.
  
  Args:
    image_paths: list of image paths.
    dir: directory in which will be stored.
  """

  try:
      os.makedirs(dir)
  except OSError as e:
      if e.errno == errno.EEXIST:
          print('Directory already exist')
      else:
          raise

  for path in image_paths:
      shutil.copy(path, os.path.join(dir,os.path.basename(path)))

def print_most_wrong_predictions(wrong_preds, n_images, image_generator, params, 
                            dir=None):
  
  """
  Prints sample of wrongfully predicted images.
  
  Args:
    wrong_preds: DataFrame of wrong classifications.
    n_images: number of images to be printed.
    image_generator: Image Generator Object.
    params: parameters used in the Image Generator.
    dir: provisional directory to store images.
  """

  # Extract Sample Images Paths
  sample_wrong_preds = wrong_preds.iloc[:n_images,]
  image_paths = sample_wrong_preds.img_path.tolist()

  # Create directory to store images

  # Make target directory
  if dir == None:
      dir = '/content/wrong_prediction_images/images'

  create_wrong_prediction_image_dir(image_paths, dir)

  # Load and process images
  data_gen = image_generator.flow_from_directory(directory=os.path.dirname(dir),
                                                 batch_size = n_images,
                                                 **params)

  # Extract images
  wrong_pred_images, _ = data_gen.next()

  # Plot images
  for i, row in enumerate(sample_wrong_preds.itertuples()):
      _, img_path, _, _, y_prob, y_true, y_pred, _ = row

      img = wrong_pred_images[i]

      plt.imshow(img[:,:,0])
      plt.title(f"{img_path}\nactual: {y_true}, pred: {y_pred} \nprob: {y_prob:.2f}")
      plt.axis(False)
      plt.show()

  # Delete provisional directory
  try:
      shutil.rmtree(os.path.dirname(dir))
  except Exception:
      print('Directory not found')

def plotrocauc(auc_list, fpr, tpr, tprs, mean_auc, mean_fpr,
               mean_tpr, title, fig, ax, path=None):
    """
    Plot the ROC curve.

    Args:
        auc_list: list of auc values.
        fpr: false positive rate list.
        tpr: true positive rate list.
        mean_auc: value.
        mean_fpr: mean fpr list.
        mean_tpr: mean tpr list.
        title: model_name.
        path: to load the data.
    """

    # Plot ROC AUC Curves
    for i in range(len(fpr)):
        ax.plot(fpr[i], tpr[i], lw = 3, alpha = 0.5,
                label='ROC fold %d (area = %0.2f)' % (i, auc_list[i]))

    # Plot diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance',
            alpha=.8)

    # Plot Mean ROC AUC
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % 
            (mean_auc, std_auc),
            lw=2, alpha=1)
    
    # Plot Confidence Interval
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', 
                    alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    # Settings
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title='ROC Curve')
    ax.legend(loc="lower right")

    # Save Figure
    filename = 'pr_roc_' + str(title) + '.jpg'
    if path != None:
        plt.savefig(os.path.join(path,filename), dpi = 100)

    plt.show()

def binary_CV(classifier, X_dict, y , k, param_dict, param_title_dictionary, 
       model_name, path):

    """ 
    Calculate cross validation metrics and ROC curves.

    Args:
        classifier: model.
        X_dict: dataset dictionary.
        y: labels.
        k: number of splits.
        param_dict: parameter dictionary for the model.
        param_title_dictionary: model hyperparameter title (String).
        model_name: self explanatory.
        path: to load the data.

    Output:
        precision_dict: precision scores per experiment dictionary.
        recall_dict: recall scores per experiment dictionary.
        f1_dict: f1 scores per experiment dictionary.
        auc_dict: auc scores per experiment dictionary.
    """

    # Desired Metrics to include
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    auc_dict = {}


    # Creating Stratified Fold for integration with CV
    skf = StratifiedKFold(n_splits = k)

    for dataset_key in X_dict:

        fig, ax = plt.subplots(1,2,figsize = (20,10), dpi = 100)

        X = X_dict[dataset_key]

        # Set Parameters
        classifier.set_params(**param_dict[dataset_key])
        
        # Experiment Title
        title = model_name + '_' + dataset_key + '_' + \
                                            param_title_dictionary[dataset_key]

        fig.suptitle(title)

        # Desired Metrics to include
        precision_list = []
        recall_list = []
        f1_list = []
        auc_list = []

        # Necessary Lists
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        tprs = []

        # Positive Rates
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, (train, test) in enumerate(skf.split(X,y)):

            # Separate in train and test samples
            X_train = X.iloc[train,:]
            y_train = y.iloc[train]
            X_test = X.iloc[test,:]
            y_test = y.iloc[test]

            # Fitting the model
            classifier.fit(X_train,y_train)

            # Predict
            y_pred = classifier.predict(X_test)

            # Find Desired Metrics and add it to list
            precision = precision_score(y_test,y_pred)
            precision_list.append(precision)
            recall = recall_score(y_test,y_pred)
            recall_list.append(recall)
            f1 = f1_score(y_test,y_pred)
            f1_list.append(f1)

            # Score function
            y_score = classifier.predict_proba(X_test)[:,1]

            # Average Precision
            average_precision = average_precision_score(y_test, y_score)

            # Plot Prediction-Recall Curve
            disp = plot_precision_recall_curve(classifier, X_test, y_test,
                                               ax = ax[0],
                                               label = 'PR Fold '+str(i)+\
                                               ' AP: {0:0.2f}'\
                                               .format(average_precision))
            ax[0].legend(loc = 'best')
            ax[0].set_title('Precision Recall Curve')

            # ROC AUC Curve
            fpr[i], tpr[i], thresholds = roc_curve(y_test,y_score)
            auc_list.append(auc(fpr[i],tpr[i]))
            aux = np.interp(mean_fpr, fpr[i], tpr[i])
            mean_tpr += aux
            mean_tpr[0] = 0.0
            tprs.append(aux)

        # Building Mean ROC AUC Curve
        mean_tpr /= k
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plotrocauc(auc_list,fpr,tpr,tprs,mean_auc,mean_fpr,mean_tpr,title,
                   fig, ax = ax[1], path=path)

        # Add lists to dictionaries
        precision_dict[dataset_key] = precision_list
        recall_dict[dataset_key] = recall_list
        f1_dict[dataset_key] = f1_list
        auc_dict[dataset_key] = auc_list

    return precision_dict, recall_dict, f1_dict, auc_dict

def storemodel(classifier, param_dict, dataset_key, X, y, path, model_name):

    """
    Stores a trained model (.joblib format) in the specified path.

    Args:
        classifier: used model to perform predictions.
        param_dict: models hyperparameter dictionary.
        dataset_key: dataset key.
        X: independent variables.
        y: dependent variable.
        path: path in which tje model is going to be stored.
        model_name: name of the model
    """ 

    # Set Parameters
    clf = classifier.set_params(**param_dict[dataset_key])

    # Train Model
    clf_fitted = clf.fit(X,y)

    # Save model
    joblib.dump(clf_fitted,path+'/'+model_name+'.joblib')


def storeresults(classifier, results, model_name, param_dict, 
                 param_title_dictionary, X, y, path):
    
    """ Creates a dataframe with tabulated results

    Args:
        classifier: used model to perform predictions.
        results: array of metric dictionaries.
        model_name: self explanatory.
        param_dict: models parameter grid.
        param_title_dictionary: model hyperparameter title (String).
        X: independent variables.
        y: dependent variables.
        path: path to load the model.

    Output: 
        df: Dataframe with results.
    """


    # Creating Dataframe that contains results
    df = pd.DataFrame({'Model':str,'Dataset':str,
                            'Mean_Precision':float,
                            'STD_Precision':float,
                            'Mean_Recall':float,
                            'STD_Recall':float,
                            'Mean_F1':float,
                            'STD_F1':float,
                            'Mean_AUC':float,
                            'STD_AUC':float},index = [0])

    for dataset_key in param_dict.keys():

        # Extract list
        precision = results[0][dataset_key]
        recall = results[1][dataset_key]
        f1 = results[2][dataset_key]
        auc = results[3][dataset_key]

        # Extract Statistic
        precision_mean = np.mean(precision)
        precision_std = np.std(precision)
        recall_mean = np.mean(recall)
        recall_std = np.std(recall)
        f1_mean = np.mean(f1)
        f1_std = np.std(f1)
        auc_mean = np.mean(auc)
        auc_std = np.std(auc)

        # Experiment model and hyperparameter
        name = model_name + '_' + param_title_dictionary[dataset_key]

        storemodel(classifier, param_dict, dataset_key, X, y, path, name)
        
        df = df.append({'Model':name,'Dataset':dataset_key,
                            'Mean_Precision':precision_mean,
                            'STD_Precision':precision_std,
                            'Mean_Recall':recall_mean,
                            'STD_Recall':recall_std,
                            'Mean_F1':f1_mean,
                            'STD_F1':f1_std,
                            'Mean_AUC':auc_mean,
                            'STD_AUC':auc_std},ignore_index=True)
        
    df.drop(index = 0, axis = 0, inplace = True)
        
    return df