import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score,confusion_matrix
from natsort import natsorted
import os 


y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]



def calculate_confusion_matrix(y_true,y_pred):
    #create labels from the ytrue
    labels = natsorted(list(set(y_true)))
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)
    
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    #
    # Print the confusion matrix using Matplotlib
    #
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), class_names = labels,show_normed = True)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
 
# calculate_confusion_matrix(y_true=y_true,y_pred=y_pred)

path = "/home/nabu/workspace/pedestrian_attribute_recog/test_dataset/custom_test_dataset_foreigners"

def generate_groundtruth(path):
    list_dirs = natsorted(os.listdir(path))

    y_true_age =[]
    y_true_gender=[]

    for each_dir in list_dirs:
        print(each_dir)

        each_dir_path = os.path.join(path, each_dir)

        age_cat,age_group,gender = each_dir.split('_')
        #lets create ground truth of each test image
        for image in os.listdir(each_dir_path):
            y_true_age.append("{}{}{}".format(age_cat,'_',age_group))
            y_true_gender.append(gender)
    return(y_true_gender,y_true_age)

gt_gender, gt_age = generate_groundtruth(path)

calculate_confusion_matrix(gt_gender,gt_gender)
calculate_confusion_matrix(gt_age,gt_age)
print(gt_gender)
print(gt_age)