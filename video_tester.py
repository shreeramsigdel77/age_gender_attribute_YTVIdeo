import pafy
import cv2
import numpy as np
import os
# import vlc
from datetime import datetime

from matplotlib import pyplot as plt

import json


from cust_detectron import ped_detectron
from cust_inference_test import deployPrediction
# url = "https://www.youtube.com/watch?v=_9OBhtLA9Ig"


filename = "vide01_det"
# record_live_youtubevideo(url,filename,True)





#dd/mm/yy H:M:S
# now.strftime("%d/%m/%Y %H:%M:%S")
# current_time = now.strftime("%H:%M:%S")
# function to add to JSON
def append_dict(filename, d):
    with open(filename, 'a+', encoding='utf-8') as fp:
        fp.write(json.dumps(d))
        fp.write("\n")

def read_list(filename):
    with open(filename, encoding='utf-8') as fp:
        return [json.loads(line) for line in fp]

def unique_elements(input_list:list):
    dictonary = {}
    if isinstance(input_list, list ):
        for item in input_list:
            dictonary[item] = dictonary.get(item,0)+1
    return dictonary

def update_dictionary(dict_updated,dict_new):
    if len(dict_new):
        for k,v in dict_new.items():
            dict_updated[k]+= dict_new[k] 
        
    
    return dict_updated


def put_text(im,text= "Hello", loc= (10,30),color = (255, 255, 255),thickness= 2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, text, loc, fontFace= font, fontScale=1, color=color, thickness= thickness, lineType = cv2.LINE_AA)
    return im

def averagePixels(img):
    r, g, b = 0, 0, 0
    count = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            tempb,tempg,tempr = img[x,y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    # calculate averages
    return (r/count), (g/count), (b/count), count

vid_path = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/output_old.avi"
# video = pafy.new(url)
# best = video.getbest()
# best = video.getbest(preftype="mp4")




capture = cv2.VideoCapture(vid_path)
grabbed, _ = capture.read()
 
#confirm 60 frames per sec
fps = capture.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

_, frame = capture.read()
h,w,c = frame.shape
scale_percent = 60 #percent of original size
# width: 1920 height: 1080
w = int(w*scale_percent/100)
h = int(h*scale_percent/100)
dim = (w,h)
interpolation = cv2.INTER_AREA
temp_sec = 0
counter = 0


#dd/mm/yy H:M:S


gender_dict_updated ={
    'Female': 0,
    'Male': 0
    }
age_dict_updated ={
    'personalLess30': 0,
    'personalLess45': 0,
    'personalLess60': 0,
    'personalLarger60': 0,
}


print("width: {} height: {}".format(w,h)) 
capture.set(cv2.CAP_PROP_BUFFERSIZE,0)

#write video_config
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(f'{filename}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))




while (capture.isOpened()):
    frame_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
    print("frame_pos",frame_pos)
    #capture frame by frame
    grabbed, frame = capture.read()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_dt = {
        "Time": now.strftime("%Y/%m/%d %H:%M:%S")
        }
    
    #red starts = 06:28:06 
    #green signals = 37:00-37:45(45sec)  39:15-40:00 45sec  41:40-42:25  45sec  44:03-44:48

    
    #  1mins 10 sec greenlight duration
    if grabbed == True:
        #lets resize abit

        frame = cv2.resize(frame,dim,interpolation)
         
        frame[:200,: ] = 0
        
        signal_section = frame[305:328,942:955]   #y+12 red
        signal_section_red = frame[308:317,942:954]
        signal_section_green = frame[318:327,942:954]

        avg_r,_,_,_ = averagePixels(signal_section_red)  #rgb
        _,avg_g,_,_ = averagePixels(signal_section_green)
        # print(avg_r,avg_g)

        # img_hsv = cv2.cvtColor(signal_section, cv2.COLOR_BGR2HSV)
        # mask1 = cv2.inRange(img_hsv, (0,50,50), (10,255,255)) #lower red mask
        # mask2 = cv2.inRange(img_hsv, (170,50,20), (180,255,255)) #upper red mask
        # # Binary mask with pixels matching the color threshold in white
        # mask = cv2.bitwise_or(mask1, mask2)

        # # Determine if the color exists on the image
        # if cv2.countNonZero(mask) > 0:
        #     print('Red is present!')
        # else:
        #     print('Green is present!')

        if avg_r>avg_g:
            # print('Red is present!')
            counter = 0
            
        else:
            # print('Green is present!')
            counter += 1
            frame_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
            # print("frame_pos",frame_pos)
            # print(counter)
        list_images,frame = ped_detectron(frame)
        if counter == 450:
            
            # print(list_images)
            # print(len(list_images))
            temp_path = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/exp_result/det_output"
            if len(list_images):
                #save image 
                for i in range(len(list_images)):
                    cv2.imwrite(os.path.join(temp_path,f"Sample_img_{i}.png"),list_images[i])

                #pedestrian model 
                pred_age_list, pred_gender_list=deployPrediction(list_images)

                # print("Age ",pred_age_list)
                # print("Age No ",len(pred_age_list))
                # print("Gender",pred_gender_list)
                # print("Gender",len(pred_gender_list))

                #counter
                gender_dict = unique_elements(pred_gender_list)
                age_dict = unique_elements(pred_age_list)

                #checking lis is empty or not
                if len(gender_dict):
                    #do something here
                    print(gender_dict)
                    print(age_dict)
                    print(current_time)
                    #update a dictionary
                    gender_dict_updated = update_dictionary(gender_dict_updated,gender_dict)
                    age_dict_updated = update_dictionary(age_dict_updated, age_dict)
                    print(gender_dict_updated)
                    print(age_dict_updated)

                    #merge dicts
                    full_dict = {
                        **current_dt,
                        **gender_dict,
                        **age_dict,
                        }

                    append_dict("result_video.jsonl",full_dict)
                    

                    
       

        # cv2.imshow("signal-color",signal_section)
        # cv2.imshow("signal-color-red",signal_section_red)
        # cv2.imshow("signal-color-green",signal_section_green)
        # cv2.waitKey(25)
        text_frame_black_img = np.zeros((150,952,3),np.uint8) 
        
        
        #time
        # put_text(frame ,loc=(w-150,30), text="{}".format(current_time),thickness=1)

        if len(gender_dict_updated):
            #gender
            put_text(text_frame_black_img ,loc=(600,30), text="{}{}".format("M:     ",gender_dict_updated["Male"]),color=(255,42,0))
            put_text(text_frame_black_img ,loc=(600,80), text="{}{}".format("F:     ",gender_dict_updated["Female"]),color=(153,0,153))

            put_text(text_frame_black_img ,loc=(600,150), text="{}{}".format("Total: ",gender_dict_updated["Male"]+gender_dict_updated["Female"]),color=(127,127,127))

            #age
            # child and youth
            put_text(text_frame_black_img ,loc=(10,30), text="{}{}".format("Age below 30:   ",age_dict_updated["personalLess30"]),color=(127,127,127))
            put_text(text_frame_black_img ,loc=(10,60), text="{}{}".format("Age 30-45:      ",age_dict_updated["personalLess45"]),color=(127,127,127))
            put_text(text_frame_black_img ,loc=(10,90), text="{}{}".format("Age 45-60:      ",age_dict_updated["personalLess60"]),color=(127,127,127))
            put_text(text_frame_black_img ,loc=(10,120), text="{}{}".format("Age 61 above:  ",age_dict_updated["personalLarger60"]),color=(127,127,127))
            


        # put_text(frame ,loc=(10,30), text="{}={}".format("Male",45))
        #updated frame
        text_frame_black_img = cv2.resize(text_frame_black_img,(500,75))

        frame[60:135,400:900]=text_frame_black_img
        cv2.imshow("Resized to {} {}".format(w,h),frame)
        

        #write video 
        out.write(frame)

        




        if cv2.waitKey(25) & 0xFF == ord('q'):  # close on q key
            break

    else:
        break
out.release()
capture.release()
cv2.destroyAllWindows()
exit()