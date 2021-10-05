# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
from datetime import datetime
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def show_img(img,windowName:str)-> None:
    cv2.imshow(windowName,img)
    
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyWindow(windowName)    

def read_img(path:str)-> np.ndarray:
    img = cv2.imread(path)
    return img

def draw_bbox(image:np.ndarray, bbox_list:list,preview:bool=False)-> np.ndarray: 
    img = cv2.rectangle(image,(int(bbox_list[0]),int(bbox_list[1])),(int(bbox_list[2]),int(bbox_list[3])),(255,0,0),2)
    if preview:
        show_img(img,"bounding box")
    return img

def crop_from_bbox(image:np.ndarray,bbox_list:list,preview:bool=False):
    cropped_image = image[int(bbox_list[1]):int(bbox_list[3]), int(bbox_list[0]):int(bbox_list[2])]
    # print(cropped_image.shape)
    if preview:
        show_img(cropped_image,"cropedWindow")
    # pass
    return cropped_image



#read img
# im = read_img("/home/nabusri/Pictures/Screenshot from 2021-09-13 06-59-01.png")
#show
# show_img(im,"Input 1")
def ped_detectron( im):
   
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    import numpy
    class_list = outputs["instances"].pred_classes.to("cpu").numpy()
    bboxes = outputs["instances"].pred_boxes
    # print(class_list)
    # print(bboxes)
    
    capture_output = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/captured_frames"
    bbox_list = []
    for i in bboxes.__iter__():
        # print(i.cpu().numpy())
        i = i.cpu().numpy()
        area = abs(i[0]-i[2]) * abs(i[1]-i[3])
        if area > 700: #filtering of area
            bbox_list.append(i)
    im_draw_bbox_copy = im.copy()
    cropped_img_list = []
   
    now = datetime.now()
    file_name_date = now.strftime("%Y-%m-%d %H:%M:%S")
    bbox_frame = im.copy()
    for lable, bbox in zip (class_list,bbox_list):
        im_copy = im.copy()
        if lable == 0:    #label 0 is person class
            # print(lable)
            # print(bbox)
            bbox_frame = draw_bbox(im_draw_bbox_copy,bbox,False)
            cropped_image = crop_from_bbox(im_copy,bbox,False)
            cropped_img_list.append(cropped_image)
   
    cv2.imwrite(os.path.join(capture_output,str(file_name_date)+"_original.png"),im)
    cv2.imwrite(os.path.join(capture_output,str(file_name_date)+"_bbox.png"),bbox_frame)
        # if lable == 9:    #label 0 is person class
        #     # print(lable)
        #     # print(bbox)
        #     draw_bbox(im_draw_bbox_copy,bbox,False)
        #     crop_from_bbox(im_copy,bbox,False)


    # #  use `Visualizer` to draw the predictions on the image.
    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # # cv2_imshow(out.get_image()[:, :, ::-1])
    # im_out = out.get_image()[:, :, ::-1]
    # #show
    # show_img(im_out,"Prediction Preview")
    
    return cropped_img_list,bbox_frame
# pedDet = PedDetectron()
# pedDet(im)
