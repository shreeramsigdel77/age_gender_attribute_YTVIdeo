import os

while 1:
    file = "/home/nabusri/BackupFiles/workspace/pedestrian_attribute_recog/resnet50_custom_pedestrian/youtube_loader.py"
    try :    
        exec(open(file).read())
        print("Restarting...")
    except:
        
        pass
    # exit()