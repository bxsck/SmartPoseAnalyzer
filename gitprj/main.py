from flask import Flask, render_template, request, session, redirect, url_for, session 
# from flask_mysqldb import MySQL
# import MySQLdb.cursors
import logging
#from flask.ext.session import Session
#from flask_sqlalchemy import SQLAlchemy
from google.cloud import storage
#import cloudstorage as gcs
import sqlalchemy
# import re
import os
import pymysql
import numpy as np
#import matplotlib.pyplot as plt
import statistics
import math
import time
import pickle
import requests
from io import BytesIO
from pandas import DataFrame
import pandas as pd
from flask import jsonify
import cv2

CLOUD_STORAGE_BUCKET = 'webapp-262307.appspot.com'

app = Flask(__name__)
app.secret_key = 'super secret key'
@app.route('/')
def index():
   return render_template('index.html')


@app.route('/upload', methods=['GET','POST'])
def upload():
    uploaded_file = request.files.get('file')
    posture = request.form.get('posture')
    gcs = storage.Client()
    if not uploaded_file:
        return render_template('upload.html')
    
    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(uploaded_file.filename)

    blob.upload_from_string(
        uploaded_file.read(),
        content_type=uploaded_file.content_type
    )
    blob.make_public()
    
    if uploaded_file :
        vdo_name = 'https://storage.googleapis.com/webapp-262307.appspot.com/{}'.format(uploaded_file.filename)
        def CreateDataFrame(X,Y,posture) :
            if posture == 'DumbellCurl' :
                Ymax = []
                Ymin = []
                
                for i in X :
                    Ymax.append(np.max(Y))
                    Ymin.append(np.min(Y))
                    
                Data = { 'frame' : X, 'angle': Y, 'max' : Ymax, 'min' : Ymin}
                df = DataFrame(Data)
                
            elif posture == 'DumbellLateralRaise' or posture == 'DumbellOverheadPress' :
                Y_L = Y[0:,0]
                Y_R = Y[0:,1]
                
                Ymax_R = []
                Ymin_R = []
                
                Ymax_L = []
                Ymin_L = []
        
                for i in X :
                    Ymax_R.append(np.max(Y_R))
                    Ymax_L.append(np.max(Y_L))
                    Ymin_R.append(np.min(Y_R))
                    Ymin_L.append(np.min(Y_L))
                L_R = abs(Y_R-Y_L)
                
                Data = { 'frame' : X, 'angle_right': Y_R, 'angle_left': Y_L, 'Max_right' : Ymax_R,'Max_left' : Ymax_L, 
                        'Min_right' : Ymin_R , 'Min_left' : Ymin_L , 'Left_Right' : L_R }
                df = DataFrame(Data)
            
            return df
                
        def analysis_angle(joint) :
            vector1 = []
            vector2 = []
                        
            vec1 = [joint[0][0] - joint[1][0], joint[0][1] - joint[1][1]]
            vector1.append(vec1)
            
            vec2 = [joint[3][0] - joint[2][0], joint[3][1] - joint[2][1]]
            vector2.append(vec2)
            
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            angle = math.degrees(np.arccos(np.clip(np.dot(vec1,vec2), -1.0, 1.0)))
            return angle
                
        def angles_posture(pose,posture) :
            if posture == 'DumbellCurl' :
                joint = [ pose[3] , pose[4] , pose[2] , pose[3] ]
                angle = analysis_angle(joint)
                return  angle
            elif posture == 'DumbellLateralRaise' :
                joint_R = [ pose[3] , pose[2] , pose[2] , pose[14]]
                angle_R = analysis_angle(joint_R)
                joint_L = [ pose[6] , pose[5] , pose[5] , pose[14]]
                angle_L = analysis_angle(joint_L)
                return  angle_L,angle_R
            elif posture == 'DumbellOverheadPress' :
                joint_R = [ pose[3] , pose[2] , pose[1] , pose[14]]
                angle_R = analysis_angle(joint_R)
                joint_L = [ pose[6] , pose[5] , pose[1] , pose[14]]
                angle_L = analysis_angle(joint_L)
                return  angle_L,angle_R
            
        def ceil_or_floor(x):
            temp = x-math.floor(x)
            if temp < 0.5 :
                x = math.floor(x)
            else :
                x = math.ceil(x)
            return x

        def scope(angle,Y):
            limit = 200
            if posture == 'DumbellCurl' :
                if abs(angle-Y) > limit :
                    return True
                else :
                    return False
            elif posture == 'DumbellLateralRaise' or posture == 'DumbellOverheadPress' :
                if abs(angle[0]-Y[0]) > limit or abs(angle[1]-Y[1]) > limit :
                    return True
                else :
                    return False

        def CheckisNaN(angle):
            if posture == 'DumbellCurl':
                if np.isnan(angle) :
                    return True
                else :
                    return False
            elif posture == 'DumbellLateralRaise' or posture == 'DumbellOverheadPress' :
                if np.isnan(angle[0]) or np.isnan(angle[1]) :
                    return True
                else :
                    return False

        def LoR_create_Xb(X):
            N = X.shape[0]
            ones = np.ones([N, 1])
            Xb = np.hstack([ones, X])
            return Xb

        def LoR_find_W_local_mul_class( X , Y , posture ):
            if posture == 'DumbellCurl':
                epoch = 10000
                lr = 0.007
            elif posture == 'DumbellLateralRaise':
                epoch = 3000
                lr = 0.0005
            elif posture == 'DumbellOverheadPress':
                epoch = 3000
                lr = 0.0005
            Xb = LoR_create_Xb(X)
            N = Xb.shape[0]
            D_1 = Xb.shape[1]
            K = Y.shape[1]
            W = np.random.randn(D_1, K)/np.sqrt(D_1)
            error_list = []
            for i in range(epoch):
                Yhat = LoR_find_Yhat_mul_class(X, W)
                error = (-Y*np.log(Yhat)).sum()
                error_list.append(error)
                S = np.dot(Xb.T, Y-Yhat)
                W = W + (lr/N)*S
            return W, error_list

        def LoR_find_Yhat_mul_class(X, W):
            Xb = LoR_create_Xb(X)
            Z = np.dot(Xb, W)
            Yhat = np.exp(Z)/np.exp(Z).sum(axis=1, keepdims = True)
            return Yhat

        def find_error_mul_class(Y, Yhat):
            N = Y.shape[0]
            Y_argmax = np.argmax(Y, axis=1)
            Yhat_argmax = np.argmax(Yhat, axis=1)
            error = 100*(Y_argmax != Yhat_argmax).sum()/N
            return error

        def create_onehot_target(label):
            K = len(np.unique(label))
            N = label.shape[0]
            onehot = np.zeros([N, K])
            for i in range(N):
                onehot[i, label[i, 0]] = 1
            return onehot

        protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [ [0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], 
                    [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

        inWidth = 368
        inHeight = 368
        threshold = 0.1
        FrameNumber = 1
        X = []
        Y = []

        cap = cv2.VideoCapture(vdo_name)
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        FrameTotal = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Video : "+str(vdo_name))
        print("Posture : "+str(posture))
        print("Total Frame : "+str(FrameTotal))

        my_FrameTotal=7
        if FrameTotal>my_FrameTotal:
            FrameCut=(FrameTotal)/my_FrameTotal

        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        Frame_select = FrameCut
        Frame = 1
        FrameError = 0

        while cv2.waitKey(1) < 0:
            ret, frame = cap.read()
            if not ret:
                cv2.waitKey()
                break
            if FrameNumber-FrameError==ceil_or_floor(Frame_select) :
                if frame.shape[1] < frame.shape[0] :
                    height, width = frame.shape[:2] # image shape has 3 dimensions
                    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
            
                    rotation_mat = cv2.getRotationMatrix2D(image_center, 90, 1.)
            
                    # rotation calculates the cos and sin, taking absolutes of those.
                    abs_cos = abs(rotation_mat[0,0]) 
                    abs_sin = abs(rotation_mat[0,1])
            
                    # find the new width and height bounds
                    bound_w = int(height * abs_sin + width * abs_cos)
                    bound_h = int(height * abs_cos + width * abs_sin)
            
                    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
                    rotation_mat[0, 2] += bound_w/2 - image_center[0]
                    rotation_mat[1, 2] += bound_h/2 - image_center[1]
            
                    # rotate image with the new bounds and translated rotation matrix
                    frame = cv2.warpAffine(frame, rotation_mat, (bound_w, bound_h))

                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]
                
                inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),(0, 0, 0), swapRB=False, crop=False)
                net.setInput(inpBlob)
                output = net.forward()

                H = output.shape[2]
                W = output.shape[3]
                
                points = []
                pose = []

                for i in range(nPoints):
                    probMap = output[0, i, :, :]

                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                
                    x = (frameWidth * point[0]) / W
                    y = (frameHeight * point[1]) / H
                    pose.append((int(x), int(y)))
                
                angle = angles_posture(pose,posture)
                
                if CheckisNaN(angle) :
                    FrameError = FrameError+1
                else :
                    if not Frame == 1 and scope(angle,Y[-1]) :
                        
                        FrameError = FrameError+1
                    else :
                        X.append(Frame)
                        Y.append(angle)
                        print("Frame number : "+str(Frame)+" \t-------------------------\t Angle : "+str(angle))
                        Frame_select = Frame_select + FrameCut
                        Frame = Frame +1  
                        FrameError = 0
            FrameNumber = FrameNumber + 1

        data = CreateDataFrame(np.asarray(X),np.asarray(Y),posture)
        if posture=='DumbellCurl' :
            col_names = [ 'frame', 'angle', 'max', 'min', 'label']
            col_data = [ 'max', 'min' ]
        elif posture=='DumbellLateralRaise' or posture == 'DumbellOverheadPress' :
            col_names = [ 'frame' , 'angle_right' , 'angle_left', 'Max_right' ,'Max_left' , 
                        'Min_right' , 'Min_left' , 'Left_Right', 'label' ]
            col_data = [  'Max_right' ,'Max_left' , 'Min_right' , 'Min_left' , 'Left_Right']

        if posture=='DumbellCurl' :
            dataset = pd.read_csv("csv/Curl7f.csv",header=None,names=col_names)
        elif posture=='DumbellLateralRaise' :
            dataset = pd.read_csv("csv/LateralRaise7f.csv",header=None,names=col_names)
        elif posture=='DumbellOverheadPress' :
            dataset = pd.read_csv("csv/OverheadPress7f.csv",header=None,names=col_names)


        predicted = []
        total_Frame = len(data['frame'])

        for i in range(1,total_Frame+1) :
            data_trian = dataset[dataset['frame']==i]
            
            X_train = np.asarray(data_trian[col_data])
            y_train = create_onehot_target(np.asarray(data_trian[['label']]))
            
            W, error_list = LoR_find_W_local_mul_class(X_train, y_train,posture)
            Yhat_train = LoR_find_Yhat_mul_class(X_train, W)

            error_train = find_error_mul_class(y_train, Yhat_train)
            
            data_predicted = data[data['frame']==i]
            X_test = np.asarray(data_predicted[col_data])

            Yhat = LoR_find_Yhat_mul_class(X_test, W)
            predicted.append(int(np.argmax(Yhat)))
        print(" ***** predicted ***** ")
        print(predicted)
        count_correct = 0
        wrong_list = []
        temp = 0
        for i in predicted :
            if i == 1 :
                count_correct = count_correct + 1
            else:
                wrong_list.append(i)
                if i == 3 :
                    temp = temp+1
        if temp > 1 :
            p = 'คุณยกแขนไม่พร้อมกัน ยกแขนให้พร้อมๆกันครับ'
        elif count_correct >= 5:
            if posture == 'DumbellCurl':
                p = 'การทำ Biceps Curl ของคุณอยู่ในรูปแบบที่ถูกต้อง พยายามรักษาความเร็วการยกในแต่ละครั้งให้คงที่จนครบ 1 เซ็ต ในระหว่างทำให้พยายามโฟกัสกล้ามเนื้อ Biceps เพื่อให้กล้ามเนื้อพยายามได้อย่างเต็มประสิทธิภาพ'
            elif posture == 'DumbellLateralRaise' :
                p = 'การทำ Latteral Raise ของคุณอยู่ในรูปแบบที่ถูกต้อง พยายามรักษาความเร็วการยกในแต่ละครั้งให้คงที่จนครบ 1 เซ็ต ในระหว่างทำให้พยายามโฟกัสกล้ามเนื้อ Biceps เพื่อให้กล้ามเนื้อพยายามได้อย่างเต็มประสิทธิภาพ'
            elif posture == 'DumbellOverheadPress' :
                p = 'การทำ Overhead Press ของคุณอยู่ในรูปแบบที่ถูกต้อง พยายามรักษาความเร็วการยกในแต่ละครั้งให้คงที่จนครบ 1 เซ็ต ในระหว่างทำให้พยายามโฟกัสกล้ามเนื้อ Biceps เพื่อให้กล้ามเนื้อพยายามได้อย่างเต็มประสิทธิภาพ'
        else:
            wrong = np.median(wrong_list)
            if posture == 'DumbellCurl':
                if wrong  == 0 :
                    p = 'คุณยกแขนน้อยเกินไป ทำให้ Range of motion ต่ำเกินไปทำให้กล้ามเนื้อ Biceps ทำงานไม่เต็มที่ พยายามยกให้มากขึ้นและพยายามเกร็งกล้ามเนื้อ Biceps เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่'
                elif wrong == 2 :
                    p = 'คุณยกแขนมากเกินไป ทำให้เกิน Range of motion ที่กล้ามเนื้อ Biceps จะทำงาน ลองยกให้ต่ำลงพยายามเกร็งกล้ามเนื้อ Biceps เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่'
            elif posture == 'DumbellLateralRaise' :
                if wrong  == 0 :
                    p = 'คุณยกแขนน้อยเกินไป ทำให้ Range of motion ต่ำเกินไปทำให้กล้ามเนื้อ Deltoid ทำงานไม่เต็มที่ พยายามยกให้มากขึ้นและพยายามเกร็งกล้ามเนื้อ Deltoid เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่'
                elif wrong == 2 :
                    p = 'คุณยกแขนมากเกินไป ทำให้เกิน Range of motion ที่กล้ามเนื้อ Deltoid จะทำงาน ลองยกให้ต่ำลงพยายามเกร็งกล้ามเนื้อ Deltoid เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่'
                elif wrong == 3 :
                    p = 'คุณยกแขนไม่พร้อมกัน ยกแขนให้พร้อมๆกันครับ'
            elif posture == 'DumbellOverheadPress' :
                if wrong  == 0 :
                    p = 'คุณยกแขนน้อยเกินไป ทำให้ Range of motion ต่ำเกินไปทำให้กล้ามเนื้อ Trapezius ทำงานไม่เต็มที่ พยายามยกให้มากขึ้นและพยายามเกร็งกล้ามเนื้อ Trapezius เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่'
                elif wrong == 2 :
                    p = 'คุณยกแขนมากเกินไป ทำให้เกิน Range of motion ที่กล้ามเนื้อ Trapezius จะทำงาน ลองยกให้ต่ำลงพยายามเกร็งกล้ามเนื้อ Trapezius เพื่อให้กล้ามเนื้อทำงานได้อย่างเต็มที่'
                elif wrong == 3 :
                    p = 'คุณยกแขนไม่พร้อมกัน ยกแขนให้พร้อมๆกันครับ'
        return render_template("/result.html",value=p)
       
    else:
        return render_template('upload.html')

@app.route('/upload.html')
def uploadfile():
    return render_template('upload.html')

@app.route('/register.html')
def register():
    return render_template('register.html')

@app.route('/login.html')
def sign():
    return render_template('login.html')

@app.route('/index.html')
def ind():
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.debug = True
    app.run(host='0.0.0.0', port=8080, debug=True)