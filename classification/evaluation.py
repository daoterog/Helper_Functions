import os
import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix, classification_report, auc

def find_most_wrong_prediction(test_data, y_true, y_pred, pred_probs, 
                                  class_names):
    
    """
    Creates a DataFrame compiling the images that were wrongfully classified by 
    the model.

    Args:
        test_data: Image Data Generator test images object
        y_true: true image labels.
        y_pred: predicted image labels.
        pred_probs: predicted probabilities.
        class_names: image labels class names.

    Output:
        wrong_preds: DataFrame with wrong prediction.
    """

    # Get the filenames of our test data
    filepaths = []
    for filepath in test_data.list_files('/content/data/test/*/*.jpg',
                                        shuffle=False):
        filepaths.append(filepath.numpy())

    # Create DataFrame
    pred_df = pd.DataFrame({
        'img_path': filepaths,
        'y_true': y_true,
        'y_pred': y_pred,
        'pred_prob': pred_probs.max(axis=1),
        'y_true_classname': [class_names[y] for y in y_true],
        'y_pred_classname': [class_names[y] for y in y_pred]
    })

    # Add column that indicates wether the prediction was right
    pred_df['pred_correct'] = pred_df.y_true == pred_df.y_pred

    # Get wrong predictions and sort them by their probability
    wrong_preds = pred_df[~pred_df.pred_correct].sort_values(by='pred_prob',
                                                                ascending=False)
    
    return wrong_preds

def across_class_results(y_true, y_pred, class_names, title, fig, ax):

    """
    Compiles the presicion, recall, and f1 scores of each class into a single 
    dataframe. It calculates the accuracy score as well.
    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        class_names: names of each target classes.
        title: title of the plot.
        fig: figure object for plotting.
        ax: axis object for plotting.
    Output:
        df_results: DataFrame compiling results for each class.
    """
    # Extract the classification report
    classification_report_dict = classification_report(y_true=y_true,
                                                    y_pred=y_pred,
                                                    output_dict=True)

    # Create empty dictionary
    class_f1_scores = {}
    class_precision = {}
    class_recall = {}

    # Create df columns
    df_columns = []
    df_values = []

    # Loop through the classification report items
    for key, value in classification_report_dict.items():
        if key == 'accuracy':
            accuracy = value
            df_columns = ['model_name',key] + df_columns
            df_values = [title, value] + df_values
            break
        else:

            # Extract Values
            name = class_names[int(Decimal(key))]
            f1_score = value['f1-score']; precision = value['precision']
            recall = value['recall']

            # Fill DataFrame Values
            df_columns.extend([name+'_f1_score', name+'_precision', name+'_recall'])
            df_values.extend([f1_score, precision, recall])

            # Construct score dictionaries for plotting
            class_f1_scores[name] = f1_score
            class_precision[name] = precision
            class_recall[name] = recall


    # Create df_results
    df_results = pd.DataFrame(df_values).transpose()
    df_results.columns = df_columns

    # Create DataFrame with dictionary
    class_scores = pd.DataFrame({"class_name": list(class_f1_scores.keys()),
                            "f1_score": list(class_f1_scores.values()),
                            'precision': list(class_precision.values()),
                            'recall': list(class_recall.values())})\
                            .sort_values("f1_score", ascending=False)

    # f1-score plot
    scores = ax.barh(range(len(class_scores)), class_scores["f1_score"].values)
    ax.set_yticks(range(len(class_scores)))
    ax.set_yticklabels(list(class_scores["class_name"]))
    ax.set_xlabel("f1-score")
    ax.set_title("F1-Scores for each Class")
    ax.invert_yaxis(); # reverse the order

    return class_scores, accuracy, df_results

def make_confusion_matrix(y_true, y_pred, accuracy, fig=None, ax=None, classes=None, 
                           norm=False, text_size=10): 

    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        fig: figure object for plotting.
        ax: axis object for plotting.
        accuracy: accuracy score calculated in across_class_results.
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).
    
    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Start Plot
    if fig == None:
        fig, ax = plt.subplots(figsize=(10,10))

    # Plot the figure and make it pretty
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax, ax=ax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title=f"Confusion Matrix, Overall Accuracy = {accuracy:.3f}",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            ax.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            ax.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

def precision_recall_barplot(class_scores, title, path):

    """
    Plots precision and recall values for each class as barplots.

    Args:
        class_scores: DataFrame that contains necessary values.
        title: of the plot
        path: to load the data.
    """

    # Make Second Plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), dpi=100)

    # precision plot
    scores = ax[0].barh(range(len(class_scores)), class_scores["precision"].values)
    ax[0].set_yticks(range(len(class_scores)))
    ax[0].set_yticklabels(list(class_scores["class_name"]))
    ax[0].set_xlabel("precision")
    ax[0].set_title("Precision Score for each Class")
    ax[0].invert_yaxis(); # reverse the order

    # recall plot
    scores = ax[1].barh(range(len(class_scores)), class_scores["recall"].values)
    ax[1].set_yticks(range(len(class_scores)))
    ax[1].set_yticklabels(list(class_scores["class_name"]))
    ax[1].set_xlabel("recall")
    ax[1].set_title("Recall Score for each Class")
    ax[1].invert_yaxis(); # reverse the order

    fig.suptitle(title)

    # Save Figure
    filename = 'pr_rc_' + str(title) + '.jpg'
    fig.savefig(os.path.join(path,filename), dpi = 100)
    plt.show()