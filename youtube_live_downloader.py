import pafy
import cv2
import numpy as np
import os


def record_live_youtubevideo(url:str,filename:str="output",preview:bool=False):
    video = pafy.new(url)
    best = video.getbest()
    
    capture = cv2.VideoCapture(best.url)
    # Check if camera opened successfully
    if (capture.isOpened() == False):
        print("Unable to read camera feed")
        print("Program Stopped")
        exit()



    #confirm 60 frames per sec
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    #convert resolution to int if they are in float

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))


    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(f'{filename}.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    capture.set(cv2.CAP_PROP_BUFFERSIZE,0)
    print("Press 'q' on the Keyboard to stop recording.")
    while (capture.isOpened()):
        # frame_pos = capture.get(cv2.CAP_PROP_POS_FRAMES)
        # print("frame_pos",frame_pos)
        #capture frame by frame
        grabbed, frame = capture.read()
    
        #  1mins 10 sec greenlight duration
        if grabbed == True:
            #write the frame
            
            out.write(frame)
        if preview:
            cv2.imshow("Preview",frame)
            
            #press q on the keyboard to stop recording
            if cv2.waitKey(25) & 0xFF == ord('q'):  # close on q key
                print("Recording Stopped!!!")
                break

        else:
            break

    #When everything is done, release the video capture and video writer objects
    capture.release()
    out.release()

    #Close all frames
    cv2.destroyAllWindows()

url = "https://www.youtube.com/watch?v=HpdO5Kq3o7Y"
filename = "vide01"
record_live_youtubevideo(url,filename,True)