from id_classification import ID_Classificaiton
from employee_detection import EE_Detection
import cv2
import torch
from PIL import Image
import numpy as np
import argparse
import os


def error(img1, img2):
    
    h_arr = []
    w_arr = []
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    h_arr.append(img1.shape[0])
    w_arr.append(img1.shape[1])
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h_arr.append(img2.shape[0])
    w_arr.append(img2.shape[1])
    
    h = max(h_arr)
    w = max(w_arr)
    
    img1 = cv2.resize(img1, (w,h), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (w,h), interpolation = cv2.INTER_AREA)


    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    msre = np.sqrt(mse)
    return mse


if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_model', type=str, default= 'yolov7.pt', help='model path')
    parser.add_argument('--video', type=str,  default = 'sample.mp4', help='video path')
    parser.add_argument('--thres', type=float, default = 0.7, help='confidence score for classification model')
    parser.add_argument('--staff_model', help='staff_model name')
    opt = parser.parse_args()


    #check if parent directory exists
    folder_arr = []
    for folder in os.listdir('runs/detect'):
        folder_arr.append(folder)

    if len(folder_arr) == 0:
        os.makedirs('runs/detect/run1')
        folder_name = 'runs/detect/run1'
    else:
        double_digit_folder = [x for x in folder_arr if len(x) >=5]
        if len(double_digit_folder) > 0:
            os.makedirs('runs/detect/'+'run'+str(int(double_digit_folder[-1][-2:])+1))
            folder_name = 'runs/detect/'+'run'+str(int(double_digit_folder[-1][-2:])+1)

        else:
            single_digit_folder = [x for x in folder_arr if len(x) <5]
            os.makedirs('runs/detect/'+'run'+str(int(single_digit_folder[-1][-1])+1))
            folder_name = 'runs/detect/'+'run'+str(int(single_digit_folder[-1][-1])+1)     

    
    #initialize classificaiton model
    staff_model = ID_Classificaiton(opt.staff_model)

    cap = cv2.VideoCapture('sample.mp4')

    count = 0
    staff = 0
    detection_model = torch.hub.load('yolov7', 'custom', path_or_model= opt.detection_model, source='local')

    images_arr = []

    while cap.isOpened():

        #read frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break

        #convert to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #count the Frame
        count+=1

        results = detection_model(frame)


        coords =  EE_Detection(thres = 0.8).detect(results, frame.shape)
        
        
        if coords:
            for count2, pts in enumerate(coords):
                im = frame[pts[1]:pts[3], pts[0]:pts[2]]

                if staff_model.output(im) > opt.thres:
                    Image.fromarray(im).save(folder_name+'/'+str(count)+'_'+str([pts[0], pts[1], pts[2], pts[3]])+'.jpg')
                    print('Staff detected on frame '+str(count)+' located at (x1,y1,x2,y2): '+str([pts[0], pts[1], pts[2], pts[3]]))
                        
        torch.cuda.empty_cache()   
                    