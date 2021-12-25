import os
import matplotlib.pyplot as plt

import tensorflow as tf

from evaluation import (find_most_wrong_prediction, across_class_results, 
                        make_confusion_matrix, precision_recall_barplot)

def plot_loss_curves(history, model_name, path):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
        history: TensorFlow model History object.
        model_name: model name.
        path: path to store plot
    """ 

    # Extract Values
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Creat Plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=100)
    fig.suptitle(model_name)

    # Plot loss
    ax[0].plot(epochs, loss, label='training_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(epochs, accuracy, label='training_accuracy')
    ax[1].plot(epochs, val_accuracy, label='val_accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    # Save Figure
    filename = 'learning_curve_' + model_name + '.jpg'
    fig.savefig(os.path.join(path, filename), dpi=100)
    plt.show

def compare_historys(original_history, new_history, initial_epochs, model_name, 
                     path):
    """
    Compares two TensorFlow model History objects.
    
    Args:
        original_history: History object from original model (before new_history)
        new_history: History object from continued model training (after 
            original_history).
        initial_epochs: Number of epochs in original_history (new_history plot 
            starts from here).
        model_name: model name.
        path: path to store plot
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=100)
    fig.suptitle(model_name)

    # Acuracy learning curve
    ax[0].plot(total_acc, label='Training Accuracy')
    ax[0].plot(total_val_acc, label='Validation Accuracy')
    ax[0].plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    ax[0].legend(loc='lower right')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('epoch')

    # Loss learning curve
    ax[1].plot(total_loss, label='Training Loss')
    ax[1].plot(total_val_loss, label='Validation Loss')
    ax[1].plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    ax[1].legend(loc='upper right')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')

    # Save Figure
    filename = 'learning_curve_' + model_name + '.jpg'
    fig.savefig(os.path.join(path, filename), dpi=100)
    plt.show()

def evaluate(model, test_data, model_name, path):

    """
    Evaluates Neural Networks performance by calculating most wrong predictions, 
    compiling accuracy, precision, recall, and f1-score, plotting confusion
    matrix, and previously mentioned scores barplots.

    Args:
        model: Neural Network.
        test_data: Image Data Generator test images object.
        model_name: name of the model
        path: path to compile experiments.

    Output:
        df_results: DataFrame with the results of each model.
        wrong_preds: DataFrame with wrong predictions.
    """

    # Make prediction with the model
    pred_probs = model.predict(test_data, verbose=1)

    # Get image labels
    y_labels = []
    for _, label in test_data.unbatch():
        y_labels.append(label.numpy().argmax())

    # Get predicted labels
    y_pred = pred_probs.argmax(axis = 1)

    # Get class names
    class_names = test_data.class_names

    # Find most wrong predictions
    wrong_preds = find_most_wrong_prediction(test_data, y_labels, y_pred, 
                                                pred_probs, class_names)

    # Start plots
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

    # Find Across Class Results
    class_scores, accuracy, df_results = across_class_results(y_labels, 
                                                    y_pred, class_names,
                                                    model_name, fig, ax[1])

    # Confusion matrix
    make_confusion_matrix(y_labels, y_pred, accuracy, fig=fig, ax=ax[0],
                               classes=class_names, norm=True)
    
    fig.suptitle(model_name)

    # Save Figure
    filename = 'cv_' + str(model_name) + '.jpg'
    fig.savefig(os.path.join(path,filename), dpi = 100)
    plt.show()

    # Make Second Plot
    precision_recall_barplot(class_scores, model_name, path)

    # Store Results
    df_results.to_csv(os.path.join(path,os.path.basename(path) + '.csv'))

    return df_results, wrong_preds

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    # Read in the image
    img = tf.io.read_file(filename)
    # Decode it into a tensor
    img = tf.io.decode_image(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img/255.
    else:
        return img

def print_most_wrong_predictions(wrong_preds, n):

    """
    Prints a sample of the most wrong predictions made.

    Args:
        wrong_preds: DataFrame with wrong predictions.
        n: number of images to print.
    """

    worng_preds_sample = wrong_preds.iloc[:n,:]

    for row in wrong_preds.itertuples():
        _, img_path, _, _, pred_prob, true_cn, pred_cn, _ = row

        img = load_and_prep_image(img_path, scale=True)
        plt.imshow(img)
        plt.title(f"{img_path}\nactual: {true_cn}, pred: {pred_cn} \nprob: {pred_prob:.2f}")
        plt.axis(False)
        plt.show()