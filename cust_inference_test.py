# from cust_pedestrain import train
import os
import pprint
from collections import OrderedDict, defaultdict
from PIL import Image

import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction
import torch
# from torch.optim import optimizer
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# from batch_engine import valid_trainer, batch_trainer
from config import argument_parser 
from dataset.AttrDataset import AttrDataset, get_transform
# from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, load_ckpt,load_state_dict
from cust_utilities import cust_load_ckp, cust_save_ckp
set_seed(605)
from torch.nn.modules.module import Module
import torch.utils.data as data

from natsort import natsorted
from torch.utils.data import dataset
import pickle


from natsort import natsorted
import cv2
from torchvision import transforms
import torchvision.transforms as T

inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

height = 256
width = 192
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


get_transform = T.Compose([
    T.Resize((height, width)),
    T.ToTensor(),
    normalize
])


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix, precision_recall_fscore_support
from mlxtend.plotting import plot_confusion_matrix

parser = argument_parser()
args = parser.parse_args()


def calculate_confusion_matrix(y_true,y_pred):
    #create labels from the ytrue
    labels = natsorted(list(set(y_pred)))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    # print(accuracy)
    
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    #
    # Print the confusion matrix using Matplotlib
    #
    
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), class_names = labels,show_normed = True)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
 
# calculate_confusion_matrix(y_true=y_true,y_pred=y_pred)


def calculate_precision_recall_fscore_support(y_true,y_pred):

    labels = natsorted(list(set(y_pred)))
    return precision_recall_fscore_support(y_true, y_pred, average=None,labels=labels)


def generate_groundtruth(path):
    
    list_dirs = natsorted(os.listdir(path))

    y_true_age =[]
    y_true_gender=[]

    for each_dir in list_dirs:
        # print(each_dir)

        each_dir_path = os.path.join(path, each_dir)

        age_cat,age_group,gender = each_dir.split('_')
        #lets create ground truth of each test image
        for image in os.listdir(each_dir_path):
            y_true_age.append("{}{}{}".format(age_cat,'_',age_group))
            y_true_gender.append(gender)
    return(y_true_gender,y_true_age)






















def flatten_nested_list(nasted_list):
    """
    input: nasted_list - this contain any number of nested lists.
    ------------------------
    output: list_of_lists - one list contain all the items.
    """

    list_of_lists = []
    for item in nasted_list:
        list_of_lists.extend(item)
    return list_of_lists






def cust_load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
    """
    load state_dict of module & optimizer from file
    Args:
        modules_optims: A two-element list which contains module and optimizer
        ckpt_file: the check point file 
        load_to_cpu: Boolean, whether to preprocess tensors in models & optimizer to cpu type
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        m.load_state_dict(sd)
    # if verbose:
    #     print("Resume from ckpt {}, \nepoch: {}, scores: {}".format(
    #         ckpt_file, ckpt['ep'], ckpt['scores']))
    # return ckpt['ep'], ckpt['scores']
    return m

def main(args,image_list,output_dir):
    Module.training = False
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    # save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        # print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    # pprint.pprint(OrderedDict(args.__dict__))

    # print('-' * 60)
    # print(f'use GPU{args.device} for testing')
   

       
    

    #test custom dataloader
    
    

    
   
    


   

   
    
    backbone = resnet50()
    classifier = BaseClassifier(nattr=5)
    model = FeatClassifier(backbone, classifier)
    model_w = torch.nn.DataParallel(model)
    
    model_w.cuda()
    # print(model_w)
    # print(model.fresh_params)
    # exit()
    param_groups = [{'params': model_w.module.finetune_params(), 'lr': args.lr_ft},
                    {'params': model_w.module.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
   


    #load the pth file
    model_path = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/exp_result/PETA/PETA1/cust_save_model/checkpoint_51.pth"
    
    # wrong prediction of overaged people
    # model_path = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/exp_result/PETA/PETA1/cust_save_model/checkpoint_99.pth"
    model, optimizer, start_epoch = cust_load_ckp(model_path,model_w,optimizer)
    
    # test_inference(model=model,test_loader= test_loader)
    # image_list = read_image_list(main_dir)
    pred_dir_gender_list,pred_dir_age_list = custom_img_loader(
        model=model,
        image_list=image_list,
        
        output_dir=output_dir
        
    )

    return pred_dir_gender_list, pred_dir_age_list


def read_image_list(path):
    image_list = natsorted(os.listdir(path))
    images = []
    for i in image_list:
        images.append(cv2.imread(os.path.join(path,i)))
    return images


def custom_img_loader(model, image_list:list,output_dir:str):
    model.eval()
    preds_probs = []
    torch_imgs = []
    pred_age_list_batch = []
    pred_gender_list_batch = []
    for img in image_list:
        #convert numpyndarray to pil image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        #convert images to tensor form
        transform_img = get_transform(img)

        #adjust the dimensions
        transform_img = transform_img[None,...]
        train_logits = model(transform_img)        
        train_probs = torch.sigmoid(train_logits)
        train_probs_temp = train_probs.detach().cpu()
        # train_probs_temp = np.around(train_probs_temp)       #problem 


        
        # print(train_probs_temp)
        
        torch_imgs = transform_img.detach().cpu()



        pred_gender_list,pred_age_list=index_to_label(train_probs_temp,torch_imgs,batch_count =0,output_dir = output_dir)

        pred_age_list_batch.append(pred_age_list)
        pred_gender_list_batch.append(pred_gender_list)
    # print(len(pred_age_list_batch))
    # print(pred_age_list_batch)
    return pred_gender_list_batch, pred_age_list_batch

def index_on_inference(image, attribute_list,labels_dict,output_dir:str ="./inference_with_index"):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #lets create blank image to write the attriutes:
    h,w,c= image.shape
    scale = 1
    if h > 150:
        target_height = 150
        scale = h/target_height
    
        image = cv2.resize(image,(int(w/scale),int(h/scale)),cv2.INTER_AREA)
    h,w,c = image.shape
    # print("height ,width ",h,w)
    blank_image = np.zeros((256,500,3), np.uint8) #y, x 
    font = cv2.FONT_HERSHEY_SIMPLEX       
    txt_position_y = 15
    txt_position_x = w +15       
    # print(attribute_list)
    # exit()
    attribute_list = attribute_list.tolist()
    # print(attr_list)
    # print(attribute_list)
    maxitem = max(attribute_list[0:4])
    #maximum value for age
    for i in range(4):
        attribute_list[i] = 1 if attribute_list[i] == maxitem else 0
    #0.5 threshold for gender
    attribute_list = np.around(attribute_list)
    
    age_group = "Unknown"
    for position,each_element in enumerate (attribute_list):
        if (position < 4) & (each_element == 1.0):
            age_group = labels_dict[position]
         
    gender = "Female" if attribute_list[4] == 0 else "Male"
        
    cv2.putText(
            img = blank_image,
            text = '{}'.format(age_group),
            org = (txt_position_x,txt_position_y),
            fontFace = font,
            fontScale = 0.5,
            color = (255,255,255),
            thickness = 1)
    txt_position_y+=15
    cv2.putText(
        img = blank_image,
        text = '{}'.format(gender),
        org = (txt_position_x,txt_position_y),
        fontFace = font,
        fontScale = 0.5,
        color = (255,255,255),
        thickness = 1)
    txt_position_y+=15




    blank_image[10:h+10,10:w+10] = image
    cv2.imwrite(os.path.join(output_dir, "inference.png"),blank_image)
    # cv2.imshow("Preview",blank_image)
    # cv2.waitKey(0)

    return gender,age_group

       
def index_to_label(pred_list:list,torch_img, batch_count:int,output_dir):
   
    labels = ['personalLess30','personalLess45','personalLess60','personalLarger60','personalMale']
    
    # print(len(labels))
    # print(len(torch_img))
    # print(type(torch_img))
    # exit()
    images = torch_img
    # print(len(images))
    # exit()
   
    for count, each_list in enumerate(pred_list):
        #converting images from tensor and previewing
        img = inv_normalize(images[count])
        prev_img = transforms.ToPILImage()(img).convert("RGB")
        cv_rgb = np.array(prev_img) #rgb next convert to bgr
        cv_bgr = cv2.cvtColor(cv_rgb,cv2.COLOR_RGB2BGR)
        # cv2.imshow("Preview",cv_bgr)
        # cv2.waitKey(0)
        # print("Each LIst check up",each_list)
        cv_bgr_copy = cv_bgr.copy()
        pred_gender, pred_age = index_on_inference(cv_bgr_copy,each_list,labels,output_dir=output_dir)
    return pred_gender, pred_age


# if __name__ == '__main__':
def deployPrediction(image_list):
        
   
    # args = parser.parse_args()
    # main_dir = "/home/nabu/workspace/pedestrian_attribute_recog/test_dataset/custom_test_dataset_foreigners/"
    # output_dir = "/home/nabu/workspace/pedestrian_attribute_recog/test_dataset/inference_custom_test_dataset_foreigners/"

    #japanese people
    # main_dir = "/home/nabu/workspace/pedestrian_attribute_recog/test_dataset/custom_test_dataset"
    # output_dir = "/home/nabu/workspace/pedestrian_attribute_recog/test_dataset/inference_custom_test_data_jap"


    # main_dir = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/test_dataset/custom_test_dataset/"
    output_dir = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/test_dataset/sample_test_results/"



    overall_pred_age_list = []
    overall_pred_gender_list = []

   
        
    pred_gender_list, pred_age_list =main(args,image_list,output_dir)
    # overall_pred_age_list.append(pred_age_list)
    # overall_pred_gender_list.append(pred_gender_list)
    return pred_age_list, pred_gender_list
    
    # # post processing convert nested list into a list
    # y_pred_age= flatten_nested_list(overall_pred_age_list)
    # y_pred_gender = flatten_nested_list(overall_pred_gender_list)

    # print(overall_pred_gender_list)
    # print(overall_pred_age_list)

    # print(len(overall_pred_gender_list))
    # print(len(overall_pred_age_list))

   

    # gt_gender, gt_age = generate_groundtruth(main_dir)

    # calculate_confusion_matrix(gt_gender,y_pred_gender)
    # result1 = calculate_precision_recall_fscore_support(gt_gender,y_pred_gender)
    # print(result1)
    # calculate_confusion_matrix(gt_age,y_pred_age)
    # result2 = calculate_precision_recall_fscore_support(gt_age,y_pred_age)
    # print(result2)
    # print(gt_gender)
    # print(gt_age)





    # os.path.abspath()

