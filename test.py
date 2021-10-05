import os
import pprint
from collections import OrderedDict, defaultdict
# from matplotlib.pyplot import axis

import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction
import torch
from torch.optim import optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, load_ckpt,load_state_dict
from cust_utilities import cust_load_ckp, cust_save_ckp
set_seed(605)


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

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    # save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for testing')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    # train_set = AttrDataset(args=args, split=args.train_split, transform=valid_tsfm)
  
    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=args.batchsize,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    # )
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    train_set = valid_set
    train_loader = valid_loader

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')
    

    labels = train_set.label
    sample_weight = labels.mean(0)

    backbone = resnet50()
    classifier = BaseClassifier(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)
    model_w = torch.nn.DataParallel(model)
    model_w.cuda()
    print(model_w)
    # print(model.fresh_params)
    # exit()
    param_groups = [{'params': model_w.module.finetune_params(), 'lr': args.lr_ft},
                    {'params': model_w.module.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
   


    #load the pth file
    model_path = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/exp_result/PETA/PETA1/cust_save_model/checkpoint_99.pth"
    # model_w_dict = torch.load(model_w_path)
    # model_w_dict = load_ckpt(model_w,model_w_path)
    model, optimizer, start_epoch = cust_load_ckp(model_path,model_w,optimizer)
       
    test_inference(model=model,train_loader=train_loader)


def test_inference(model, train_loader):
    train_probs,torch_imgs = batch_tester(
        model=model,
        train_loader=train_loader,
        
    )
    # # print(type(train_probs))
    # train_probs = np.around(train_probs)
    # index_to_label(train_probs,torch_imgs)

import cv2
from torchvision import transforms
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)


def batch_tester(model, train_loader):
   
    preds_probs = []
    torch_imgs = []
    # lr = optimizer.param_groups[1]['lr']

    for count, (imgs, gt_label, imgname) in enumerate(train_loader):
       
    
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        
        train_logits = model(imgs)
        
        # #converting images from tensor and previewing
        # images = imgs.cpu()
        # print(len(images[0]))
        # print(images.shape)
        # images[0] = inv_normalize(images[0])
        # prev_img = transforms.ToPILImage()(images[0]).convert("RGB")
        # prev_img.show()
        # print(prev_img.size)
        # exit()
        # train_loss = criterion(train_logits, gt_label)

        # train_loss.backward()
        
        train_probs = torch.sigmoid(train_logits)
        # preds_probs.append(train_probs.detach().cpu().numpy())
        # torch_imgs.append(imgs.detach().cpu().numpy())

        # print(type(train_probs))
        gt_label_temp = gt_label.detach().cpu()
        train_probs_temp = train_probs.detach().cpu()

        # print(train_probs_temp)
              

        # train_probs_temp_2 = np.around(train_probs_temp)   #threshold 0.5
       

        torch_imgs = imgs.detach().cpu()
        index_to_label(gt_label_temp,train_probs_temp,torch_imgs,batch_count = count)

        # print(preds_probs)
        if count == 5:
            # print(gt_label)
            break
        

    # gt_label = np.concatenate(gt_list, axis=0)
    # preds_probs = np.concatenate(preds_probs, axis=0)
    return preds_probs, torch_imgs

def index_on_inference(image, attribute_list,labels_dict,pre_filename,output_dir:str ="./inference_with_index"):
    
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
    print(attribute_list)
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
    cv2.imwrite(os.path.join(output_dir, pre_filename+"inference.png"),blank_image)
    cv2.imshow("Preview",blank_image)
    cv2.waitKey(0)



    # cv2.imshow("Preview",blank_image)
    # cv2.imwrite(os.path.join(args.inference_dir,"inference"+img_path.split('/')[-1]),blank_image)



def index_to_label(gt_label_temp,pred_list,torch_img, batch_count):
    # print(len(pred_list))
    set1 = ['accessoryHat','accessoryMuffler','accessoryNothing','accessorySunglasses','hairLong'] #5 (0-4)
    set2 = ['upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt','upperBodyOther','upperBodyVNeck'] #10 (5-14)
    set3 = ['lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt','lowerBodyTrousers'] # 6 (15-20)
    set4 = ['footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker'] #4 (21-24)
    set5 = ['carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags'] #5 (25-29)
    set6 = ['personalLess30','personalLess45','personalLess60','personalLarger60'] #4 (30-33)
    set7 = ['personalMale']
    labels = set1+set2+set3+set4+set5+set6+set7

    labels = set6+set7
    
    # print(len(labels))
    # print(len(torch_img))
    # print(type(torch_img))
    # exit()
    images = torch_img
    # print(len(images))
    # exit()
   
    for count, each_list in enumerate(pred_list):
        # print("Ground Truth ", gt_label_temp[count])
        # print("Prediction", each_list)
        
        #  Image preview
        #converting images from tensor and previewing
        
        img = inv_normalize(images[count])
        prev_img = transforms.ToPILImage()(img).convert("RGB")
        cv_rgb = np.array(prev_img) #rgb next convert to bgr
        cv_bgr = cv2.cvtColor(cv_rgb,cv2.COLOR_RGB2BGR)
        # cv2.imshow("Preview",cv_bgr)
        # cv2.waitKey(0)
        cv_bgr_copy = cv_bgr.copy()
        index_on_inference(cv_bgr_copy,each_list,labels,"{}{}".format(batch_count,count))
        
        
       

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

    # os.path.abspath()

"""
载入的时候要：
from tools.function import LogVisual
sys.modules['LogVisual'] = LogVisual
log = torch.load('./save/2018-10-29_21:17:34trlog')
"""
