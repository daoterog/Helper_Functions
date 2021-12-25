import os
import shutil
import zipfile
import errno
import random

import numpy as np

def split_images(root='data/', train_size=0.7, test_proportion=0.5):
    """
    Splits the images into train and test sets. This allows us to perform
    certain analysis after evaluating the model.

    Args:
        root: The root path in which is stored the data
        train_size: proportion of the data that will correspond to the training
            set.
        test_proportion: proportion of the remaining data (after splitting the 
            training portion) that will correspond to the test set (eg. if 
            train_size=0.7 then the test_size will be of 0.15 because 
            (1-0.7)*0.5=0.15)
    """

    train_path = os.path.join(root,'train')
    validation_path = os.path.join(root,'validation')
    test_path = os.path.join(root,'test')

    # Delete directories to avoid over-riding shuffle
    try:
        shutil.rmtree(train_path)
        shutil.rmtree(validation_path)
        shutil.rmtree(test_path)
    except Exception:
        print('Not found')

    # Create train and test directories
    os.mkdir(train_path)
    os.mkdir(validation_path)
    os.mkdir(test_path)
    
    for category in os.listdir(root):
        if category == 'test' or category == 'train' or category == 'validation':
            pass
        else:
            # Extract the image paths
            category_path = os.path.join(root,category)
            category_images = os.listdir(category_path)

            # Randomly choose train images
            category_train_images = random.choices(category_images, 
                                                    k=int(train_size * \
                                                          len(category_images))) 

            # Convert lists to set in order to get the remaining image paths
            category_set = set(category_images)
            train_set = set(category_train_images)
            category_validation_test_images = list(category_set - train_set)

            # Randomly choose validation images
            category_validation_images = random.choices(category_validation_test_images,
                                                         k=int((1-test_proportion) *
                                                               len(category_validation_test_images)))
            
            # Convert lists to sets in order to get the remaining image paths
            category_validation_test_images_set = set(category_validation_test_images)
            validation_set = set(category_validation_images)
            category_test_images = list(category_validation_test_images_set -
                                        validation_set)

            # Create category directory
            category_train_path = os.path.join(train_path,category)
            category_validation_path = os.path.join(validation_path,category)
            category_test_path = os.path.join(test_path,category)
            try:
                os.mkdir(category_train_path)
                os.mkdir(category_validation_path)
                os.mkdir(category_test_path)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    print('Directory already exist')
                else:
                    raise
            
            # Copy train files
            for image in category_train_images:
                src_path = os.path.join(category_path, image)

                if os.path.isfile(src_path):
                    dst_path = os.path.join(category_train_path,image)
                    shutil.copy(src_path, dst_path)

            # Copy validation files
            for image in category_validation_images:
                src_path = os.path.join(category_path, image)

                if os.path.isfile(src_path):
                    dst_path = os.path.join(category_validation_path,image)
                    shutil.copy(src_path, dst_path)
            
            # Copy test files
            for image in category_test_images:
                src_path = os.path.join(category_path, image)

                if os.path.isfile(src_path):
                    dst_path = os.path.join(category_test_path,image)
                    shutil.copy(src_path, dst_path)

def bring_data(path_list, root='data'):
    
    """
    Brings zip folders and unzips them into the specified root. 

    Output:
        path_list: list of paths to each of the compressed folders stored
            in the drive.
        root: path in which the compressed folder will be unzipped.
    """

    # Delete directories to avoid over-riding files
    try:
        shutil.rmtree(root)
    except Exception:
        print('Not found')

    # Create root directory
    os.mkdir(root)
    
    # Copying zip files
    for filepath in path_list:
        
        # Extract Category Names
        category = os.path.basename(filepath)

        # Copy zip files
        shutil.copy(filepath,root)

        # Unzip the files
        local_zip_folder = os.path.join(root,category)
        with zipfile.ZipFile(local_zip_folder, 'r') as zip_ref:
            zip_ref.extractall(root)

        # Remove zip folder
        os.remove(local_zip_folder)

def copytree(dir, src, dst, symlinks=False, ignore=None):
    """ 
    Used to copy a whole directory into a destination.

    Args:
        dir: name of the directory.
        src: source path.
        dst: destination path. 

    """

    dst = os.path.join(dst,dir)

    try:
        os.mkdir(dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exist')
        else:
            raise

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def copy_folder(path, destination):

    """
    Copies the experiment directory into the specifified destination. It allows
    the user to fill a description in order to give further insight of the 
    experimentation.

    Args:
        path: path to the experiment folder.
        destination: path to the destination folder.
    """

    comment = str(input('Enter the experiment description:'))

    # Extract experiment name
    experiment = os.path.basename(path)

    np.savetxt(os.path.join(path,experiment + '.txt'),[comment], fmt = '%10s')

    copytree(experiment,path,destination)