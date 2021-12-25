import os
import cv2

import scipy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
import tensorflow as tf

def extract_edges(img):

    """
    Extracts the edges of the image using the canny algorithm.

    Args:
        img: image.

    Outputs:
        edged: highlighted edges image.
    """

    # Blurring 
    blurred = cv2.bilateralFilter(img,15,150,150)

    # Edge Detection
    v = np.median(blurred)
    sigma = 0.33

    lower = int(max(0,(1-sigma)*v))
    upper = int(min(255,(1+sigma)*v))

    img = np.uint8(img)

    edged = cv2.Canny(img,lower,upper)

    return edged

def make_cut(img, edged):

    """
    Cuts the image on the corresponding axis.

    Args:
        img: image.
        edged: highlighted edges image.

    Output:
        img: cutted image.
        edged: cutted highlighted edges image.
    """

    index = 0
    for i in range(edged.shape[0]):
        aux = np.sum(edged[i])
        if aux != 0:
            index = i
            break

    if index != 0:
        img = img[index-1:]
        edged = edged[index-1:]
    
    return img, edged

def center_image_IDG(img):

    """
    Centers the image taking into account the borders of the object. This method 
    is created specifically for using it along the tensorflow's 
    ImageDataGenerator. It is significantly faster to use. 

    Args:
        img: image that is going to be centered.
        
    Outputs:
        img: centered image.
        edged: centered image with highlighted borders.
    """

    img = tf.keras.preprocessing.image.img_to_array(img)
    
    # Extract edges
    edged = extract_edges(img)

    # Superior cut
    img_cutted, edged_cutted = make_cut(img, edged)

    # Square the image
    max_img_shape = np.max(tf.shape(img_cutted).numpy())
    new_image_shape = (max_img_shape, max_img_shape)
    img_cutted = tf.image.resize(img_cutted, list(new_image_shape))
    edged_cutted = cv2.resize(edged_cutted, new_image_shape, 
                              interpolation = cv2.INTER_AREA)

    # Transpose it
    img_trans = tf.squeeze(img_cutted.numpy().transpose())
    edged_trans = edged_cutted.transpose()
    
    # Left cut
    img_trans_cutted, edged_trans_cutted = make_cut(img_trans, edged_trans)

    # Flip it
    edged_trans_flip = np.flip(edged_trans_cutted)
    img_trans_flip = np.flip(img_trans_cutted)

    # Right cut
    img_trans_flip_cutted, edged_trans_flip_cutted = make_cut(img_trans_flip, 
                                                              edged_trans_flip)
    
    # Square the image
    max_img_shape = np.max(tf.shape(img_trans_flip_cutted).numpy())
    new_image_shape = (max_img_shape, max_img_shape)
    img_trans_flip_cutted = cv2.resize(np.float32(img_trans_flip_cutted), new_image_shape, 
                              interpolation = cv2.INTER_AREA)
    edged_trans_flip_cutted = cv2.resize(edged_trans_flip_cutted, new_image_shape, 
                              interpolation = cv2.INTER_AREA)

    # Transpose it 
    edged_trans_flip_trans = edged_trans_flip_cutted.transpose()
    img_trans_flip_trans = img_trans_flip_cutted.transpose()

    # Inferior cut
    img_trans_flip_trans_cutted, edged_trans_flip_trans_cutted = make_cut(img_trans_flip_trans,
                                                            edged_trans_flip_trans)
    
    img1 = np.flip(img_trans_flip_trans_cutted.transpose()).transpose()
    edged1 = np.flip(edged_trans_flip_trans_cutted.transpose()).transpose()

    # Square the image
    max_img_shape = np.max(tf.shape(img1).numpy())
    new_image_shape = (max_img_shape, max_img_shape)
    img = cv2.resize(np.float32(img1), new_image_shape, 
                              interpolation = cv2.INTER_AREA)

    img = tf.expand_dims(img, axis=2)

    return img

def center_image(img):

    """
    Centers the image taking into account the borders of the object.

    Args:
        img: image that is going to be centered.

    Outputs:
        img: centered image.
        edged: centered image with highlighted borders.
    """

    edged = extract_edges(img)

    # Superior cut
    img, edged = make_cut(img, edged)

    edged_trans = edged.transpose()
    img_trans = img.transpose()

    # Left cut
    img_trans, edged_trans = make_cut(img_trans, edged_trans)

    edged_trans_flip = np.flip(edged_trans)
    img_trans_flip = np.flip(img_trans)

    # Right cut
    img_trans_flip, edged_trans_flip = make_cut(img_trans_flip, edged_trans_flip)

    edged_trans_flip_trans = edged_trans_flip.transpose()
    img_trans_flip_trans = img_trans_flip.transpose()

    # Inferior cut
    img_trans_flip_trans, edged_trans_flip_trans = make_cut(img_trans_flip_trans,
                                                            edged_trans_flip_trans)
    
    img = np.flip(img_trans_flip_trans.transpose()).transpose()
    edged = np.flip(edged_trans_flip_trans.transpose()).transpose()

    return img, edged

def preprocessing(img, resize, blur, grayscale, rescale, edges, center):

    """
    Applies the specified functions to the image.

    Args:
        img: image to be preprocessed.
        resize: reference size to give to the image.
        blur: Boolean marker that indicates to blur the image.
        grayscale: Boolean marker that indicates to convert the image to 
            grayscale.
        recale: Boolean marker that indicates to rescale image pixel values.
        edges: Boolean marker that indicates to extract edges of the image.
        center: Boolean marker that indicates to center the image.

    Output:
        img: preprocessed img
    """
    
    # Convert it to GrayScale or to RGB
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('metal_grayscale.jpg',img)

        # Recale Values between 0-1
        if rescale and not edges:
            img_res = img/255
            # cv2.imwrite('metal_rescaled.jpg', img_res)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if center:
        img, edged = center_image(img)
        # cv2.imwrite('metal_centered.jpg', img)
        # cv2.imwrite('metal_edges.jpg', edged)
        if edges:
            img = edged

    # Edges and blur
    if edges and not center:
        img = extract_edges(img)
    elif blur:
        img = cv2.bilateralFilter(img,15,50,150)
        # cv2.imwrite('metal_blurred.jpg', img)

    # Resize it to certain dimensions
    if resize[0] != 0 and resize[1] != 1:
        img = cv2.resize(img, resize, interpolation = cv2.INTER_AREA)
        # cv2.imwrite('metal_resized.jpg', img)

    return img

def load_images(root = 'data', resize = (0,0), blur = False, grayscale = False,
                rescale = False, edges = False, center = False):
    
    """
    This function reads the images and stores them into a dictionary.

    Args:
        root: path to the data/image directory
        resize: reference size to give to the image.
        blur: Boolean marker that indicates to blur the image.
        grayscale: Boolean marker that indicates to convert the image to 
            grayscale.
        recale: Boolean marker that indicates to rescale image pixel values.
        edges: Boolean marker that indicates to extract edges of the image.
        center: Boolean marker that indicates to center the image.

    Output:
        images: dictionary of images with each label as key of the dictionary.

    """
    
    with tf.device('/device:GPU:0'):

        images = {}

        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            category = []

            for image in os.listdir(label_path):
                # Read Image
                img = cv2.imread(os.path.join(label_path, image))
                img = np.float32(img)

                img = preprocessing(img, resize, blur, grayscale, rescale, 
                                    edges, center)

                if img is not None:
                    category.append(img)

            images[label] = category

    return images

def load_sample_images(root, resize = (0,0), blur = False, grayscale = False,
                rescale = False, edges = False, center = False):
    
    """
    Reads specified image and plots the raw and preprocessed version of it.

    Args:
        root: path to the image.
        resize: reference size to give to the image.
        blur: Boolean marker that indicates to blur the image.
        grayscale: Boolean marker that indicates to convert the image to 
            grayscale.
        recale: Boolean marker that indicates to rescale image pixel values.
        edges: Boolean marker that indicates to extract edges of the image.
        center: Boolean marker that indicates to center the image.

    Outputs:
        img: preprocessed image.
    """
    
    # Read Image
    img = cv2.imread(root)

    # Plot image before processing
    plt.imshow(img)
    plt.title('Raw Image')
    plt.show()
    img = np.float32(img)

    img = preprocessing(img, resize, blur, grayscale, rescale, edges, center)

    # Plot image after processing
    plt.imshow(img)
    plt.title('Processed Image')
    plt.show()

    return img

def extract_patches(img, patch_size, step_size, include_empty_patches=False):

    """
    Extracts patches of each image, flattens it, and adds it to a list.

    Args:
        img: image.
        patch_size: size of the squared patch.
        step_size: specified step of the mask that will run through the image.
        include_empty_patches: Boolean for including patches with missing 
            information.

    Output:
        patches: list flattened image patches.
    """

    with tf.device('/device:GPU:0'):
        patches = []
        for y in range(0, img.shape[0]-patch_size+1, step_size):
            for x in range(0, img.shape[1]-patch_size+1, step_size):
                patch = img[y:y+patch_size,x:x+patch_size]
                if patch.shape==(patch_size, patch_size) and \
                                    (include_empty_patches or np.sum(patch)!=0):
                    patches.append(patch)
    return patches

def get_visual_dictionary(X, patch_size, step_size, dict_size):

    """
    Creates dictionary of specified size of features using the KMeans algorithm.

    Args:
        X:
        patch_size: size of the squared patch.
        step_size: specified step of the mask that will run through the image.
        dict_size: size of the dictionary (number of clusters created by the 
            algorithm).

    Outputs:
        km.cluster_centers_: visual dictionary of features.
    """

    with tf.device('/device:GPU:1'):
        patches = []
        for img in X:
            patches += [i.reshape(patch_size**2) for i in extract_patches(img, 
                                                        patch_size, step_size)]

        cinit = np.zeros((dict_size, patch_size**2))
        km = KMeans(n_clusters=dict_size, init=cinit, n_init=1)
        km.fit(patches)
        return km.cluster_centers_

def get_closest(patch, dictionary):

    """
    Get the closes feature to the one identified in a patch.

    Args:
        patch: patch of the image.
        dictionary: visual dictionary of features.

    Outputs:
        r: index of the identified feature in the dictionary.
    """

    with tf.device('/device:GPU:2'):
        dmin, r = np.inf, None
        for i, vw in enumerate(dictionary):
            distance = scipy.linalg.norm(patch-vw)
            if distance<dmin:
                dmin = distance
                r = i
    return r

def get_histogram(img, patch_size, step_size, dictionary):

    """
    Creates the feature vector (histogram) of each image.

    Args:
        img: image to get the dictionary.
        patch: patch of the image.
        step_size: specified step of the mask that will run through the image.
        dictionary: visual dictionary of features.

    Output:
        np.array(h)*1./np.sum(h): normalized histogram.
    """

    with tf.device('/device:GPU:3'):
        patches = [i.flatten() for i in extract_patches(img, patch_size, step_size)]
        
        vws = np.array([get_closest(patch, dictionary) for patch in patches])
        h = [np.sum(vws==i) for i in range(len(dictionary))]

    return np.float64(np.array(h)*1./np.sum(h))

def applypca(X):

    """
    Applies PCA to the dataset.

    Args:
        X: dataset

    Outputs:
        X_pca: pca dataset.
        pca: principal components analysis object.
    """

    pca = PCA()
    X_pca = pca.fit_transform(X)
    X_pca = pd.DataFrame(X_pca)

    return X_pca, pca

def applynmf(X, cont):

    """
    Applies NMF to the dataset.

    Args:
        X: dataset

    Outputs:
        X_nmf: nmf dataset.
        nmf: Non-negative Matrix Function object.
    """

    nmf = NMF(n_components = cont)
    X_nmf = nmf.fit_transform(X)
    X_nmf = pd.DataFrame(X_nmf)

    return (X_nmf,nmf)
