import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import keras
from keras import backend as K
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import clone_model
from keras.callbacks import EarlyStopping
# from yolo_utils import *
from darkflow.net.build import TFNet
import re
import time
from sklearn.cluster import DBSCAN
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

class darkflow_prediction():

    def __init__(self):
        self.options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.5}
        self.tfnet = TFNet(self.options)
        self.cluster = []
        self.FRAME_BUFFER = 5
    def image(self, image_file):
        self.image = cv2.imread(image_file, 1)
        self.result = self.tfnet.return_predict(self.image)
        print(self.result)
        self.print_box()
        cv2.waitKey(0)

    def print_box(self, frame_result, frame_image):
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(frame_result)):
            coordtl = (frame_result[i]['topleft']['x'], frame_result[i]['topleft']['y'])
            coordbr = (frame_result[i]['bottomright']['x'], frame_result[i]['bottomright']['y'])
            cv2.rectangle(frame_image, coordtl, coordbr, (0,255,0), 2)
            s = str(frame_result[i]['label'] + ": " + str(frame_result[i]['confidence']))
            text_coord = (coordtl[0], coordtl[1]-10)
            cv2.putText(frame_image, s, text_coord, font, 1, (250,250,0))
        #cv2.imshow("memes", frame_image)

    def print_box_with_clusters(self, asd):
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(self.result)):
            coordtl = (self.result[i]['topleft']['x'], self.result[i]['topleft']['y'])
            coordbr = (self.result[i]['bottomright']['x'], self.result[i]['bottomright']['y'])
            cv2.rectangle(self.image,coordtl,coordbr,(0,255,0),2)
            s = str(self.result[i]['label'] + ": " + str(self.result[i]['confidence']))
            text_coord = (coordtl[0], coordtl[1]-10)
            cv2.putText(self.image, s, text_coord, font, 1, (250,250,0))
        for i, val in enumerate(asd):
            cv2.putText(self.image, str(val[1]), tuple(val[0]), font, 1, (250,250,0), 3)
        #cv2.imshow("memes", self.image)

    def video(self, video_file):
        self.video = cv2.VideoCapture(video_file)
        self.images, interm, count = [], [], 1

        # Results: list of lists of object dictionaries [frame1[{object1}, {object2}, {object3}], frame2[{}, {}, {}]]
        self.video_results_full = []

        # Same as above, but results are split into groups of 5
        self.video_results_split = []

        # Grand boxes for each group of 5 frames
        self.group_grand_boxes = []

        # Actual trajectories of objects through frames
        self.object_trajectories = {}

        try:
            while self.video.isOpened():
                # Read 1 frame of the video (as image), append to our list of images
                ret, frame_image = self.video.read()
                self.images.append(np.copy(frame_image))

                # Run YOLO to get the detected objects for this particular frame
                frame_result = self.tfnet.return_predict(frame_image)
                self.print_box(frame_result, frame_image)
                frame_result = self.refactor_result(frame_result)
                self.video_results_full.append(frame_result)
                interm.append(frame_result)

                # Use a sliding window approach to compute groups/grand boxes
                if count > self.FRAME_BUFFER:
                    interm.pop(0)
                if len(interm) == self.FRAME_BUFFER:
                    self.video_results_split.append(interm[:])
                    grand_boxes = self.get_clusters(self.video_results_split[-1])
                    if not grand_boxes:
                        continue
                    self.print_grand_box(grand_boxes)
                    self.group_grand_boxes.append(grand_boxes)
                    if len(self.group_grand_boxes) >= 2:
                        self.track_objects_between_frames(self.group_grand_boxes[-2], self.group_grand_boxes[-1])
                count += 1
                #cv2.waitKey(1)
        except AssertionError:
            pass

        #print("GROUP GRAND BOXES: ", self.group_grand_boxes)
        for group in self.group_grand_boxes: # group is a list with dictionaries
            for obj in group: # obj is a dictionary
                if "prev" not in obj: # prev is the string key in the dictionary
                    self.object_trajectories[self.hash_object(obj)] = [obj]
                else:
                    #print(self.object_trajectories[obj['prev']])
                    self.object_trajectories[obj['prev']].append(obj)
                    self.object_trajectories[self.hash_object(obj)] = self.object_trajectories[obj['prev']]
                    #self.object_trajectories[obj['prev']] = None
        self.object_trajectories = {k: v for k, v in self.object_trajectories.items() if v is not None}
        for i in self.object_trajectories:
            pass
            #print("\n\n")
            #print(self.object_trajectories[i])

        # trainingSize = int(round(0.75 * len(self.object_trajectories)))
        # trainingSet = [self.object_trajectories[i] for i in range(trainingSize)]
        # testSet = [self.object_trajectories[i] for i in range(trainingSize, len(self.object_trajectories))]
        dataIn, dataOut = self.getXYAll(self.object_trajectories)
        #trainX = []
        #trainY = []
        ''''testX = []
        testY = []
        for i in range(len(dataX)): #length is the number of objects we have in frame, same for both X and Y
            pathX = dataX[i]
            pathY = dataY[i]
            trainingSize = int(round(0.75 * len(pathX)))
            trainX.extend([[pathX[i]] for i in range(trainingSize)])
            testX.extend([[pathX[i]] for i in range(trainingSize, len(pathX))])
            [trainY.append(pathY[i]) for i in range(trainingSize)]
            [testY.append(pathY[i]) for i in range(trainingSize, len(pathX))]

        look_back = 1
        trainX = np.array(trainX)
        testX = np.array(testX)
        trainY = np.array(trainY)
        testY = np.array(testY)

        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))'''

       # create and fit the LSTM network
        print("DATA IN (OBJECT DATA) SHAPE: ", dataIn.shape)
        print("DATA OUT (OBJECT OUTPUT) SHAPE: ", dataOut.shape)
        model = Sequential()
        model.add(LSTM(200, input_shape=(5, 2)))
        model.add(Dense(2))
        model.compile(loss='mean_squared_error', optimizer=Adam(decay=0.0001))
        model.fit(dataIn, dataOut, epochs=200, batch_size=1, verbose=2)
        model.save(video_file[3:])
        # make predictions
        trainPredict = model.predict(dataIn)
        #testPredict = model.predict(testX)

        #print("TRAIN PREDICT: ", trainPredict)

    """
    def separateXandY(self, object_trajectories):
        #print("OBJECT TRAJECTORIES")
        #print(object_trajectories)
        objectDataX, objectDataY = [], []
        for object in object_trajectories:
            #print("OBJECT " + object)
            dataX, dataY = [], []
            for point in object_trajectories[object]:
                #print("POINT")
                #print(point)
                dataX.append(point["x"])
                dataY.append(point["y"])
            objectDataX.append(dataX)
            objectDataY.append(dataY)
        print(objectDataX)
        print(objectDataY)
        return np.array(objectDataX), np.array(objectDataY)
    """

    def getXYAll(self, object_trajectories):
        objectData = []
        objectOutput = []
        #print("OBJECT TRAJECTORIES: ", object_trajectories)
        for obj in object_trajectories:
            #print("LEN OBJECT TRAJ OF OBJ: ", len(object_trajectories[obj]))
            for i in range(0, len(object_trajectories[obj])-5):
                data = []
                for j in range(i, i+5):
                    data.append([object_trajectories[obj][j]["x"], object_trajectories[obj][j]["y"]])
                objectOutput.append([object_trajectories[obj][i+5]["x"], object_trajectories[obj][i+5]["y"]]) # next position after 5 frames
                objectData.append(data[:]) # append single sample, data = an instance of 5 time steps
        #print('OBJECT DATA: ', objectData) # [[[x, y], [x, y], [x, y], [x, y], [x, y]],
                                        # [[x, y], [x, y], [x, y], [x, y], [x, y]]]
        #print('OBJECT OUTPUT: ', objectOutput) # output = where is the next location

        #print('objectOutput', objectOutput) # output matches samples [[x, y], [x, y], [x, y], [x, y], [x, y]]
        return np.array(objectData), np.array(objectOutput)

    def hash_object(self, detected_object):
        return str(detected_object["x"]) + str(detected_object["y"]) + str(detected_object["class"]) + str(detected_object["confidence"])

    def refactor_result(self, result):
        new_result = []
        for elem in result:
            coordtl = (elem['topleft']['x'], elem['topleft']['y'])
            coordbr = (elem['bottomright']['x'], elem['bottomright']['y'])
            classif = elem['label']
            confidence = elem['confidence']
            width = coordbr[0] - coordtl[0]
            height = coordbr[1] - coordtl[1]
            x_center = coordtl[0] + width//2
            y_center = coordtl[1] + height//2
            new_elem = {'x': x_center, 'y': y_center, 'width': width,
                        'height': height, "class": classif, "confidence": confidence}
            new_result.append(new_elem)
        return new_result

    def get_clusters(self, group):
        cluster_points = [] #still need all the individual point data to get individual box data for box averaging
        for frame in group: #each frame object is 5 video frames
            for object_det in frame:
                cluster_points.append([object_det['x'], object_det['y']]) #extracts the coordinates of each object
        #print("cluster_points", cluster_points)
        if cluster_points == []:
            return None
        model = DBSCAN(eps=100, min_samples=2).fit(np.array(cluster_points))
        clusters = [(cluster_points[i], model.labels_[i]) for i in range(len(cluster_points))]
        clustered_points = {}
        for point in clusters:
            if point[1] not in clustered_points:
                clustered_points[point[1]] = []
            clustered_points[point[1]].append((point[0][0], point[0][1]))
        #print(clustered_points)
        grand_boxes = [] #creating each single grand box
        for cluster_val in clustered_points:
            avgd_x, avgd_y = 0, 0
            for point in clustered_points[cluster_val]:
                avgd_x += point[0]
                avgd_y += point[1]
            avgd_x /= len(clustered_points[cluster_val])
            avgd_y /= len(clustered_points[cluster_val])
            width, height = 0, 0
            classif, confidence = '', ''
            for point in clustered_points[cluster_val]:
                for frame in group:
                    for object_det in frame:
                        if object_det['x'] == point[0] and object_det['y'] == point[1]:
                            width += object_det['width']
                            height += object_det['height']
                            classif = object_det['class']
                            confidence = object_det['confidence']
            width /= len(clustered_points[cluster_val])
            height /= len(clustered_points[cluster_val])
            box = {'x': avgd_x, 'y': avgd_y, 'width': width,
                   'height': height, "class": classif, "confidence": confidence}
            grand_boxes.append(box) #creating a grand box for each object for every 5 frames
        #print("grand boxes", grand_boxes)
        return grand_boxes

    def print_grand_box(self, grand_boxes):
        x_points = [box['x'] for box in grand_boxes]
        y_points = [box['y'] for box in grand_boxes]
        image = self.images[-1].copy()
        for i in range(len(x_points)):
            cv2.circle(image, (int(x_points[i]), int(y_points[i])), 15, (255-int(255*i//len(x_points)),0,int(255*i//len(x_points))), -1)
        #cv2.imshow("centroids", image)
        #cv2.waitKey(2)

    def track_objects_between_frames(self, curr_frame, next_frame):
        for grand_object in curr_frame:
            closest_dist = 999999999999
            closest_obj, closest_obj_idx = None, None
            for next_object_idx in range(len(next_frame)):
                next_object = next_frame[next_object_idx]
                euclidean_dist = np.sqrt((next_object['x'] - grand_object['x'])**2 + (next_object['y'] - grand_object['y'])**2)
                if euclidean_dist < closest_dist:
                    closest_dist = euclidean_dist
                    closest_obj = next_object
                    closest_obj_idx = next_object_idx
            if closest_dist <= 25:
                grand_object['next'] = self.hash_object(closest_obj)
                next_frame[closest_obj_idx]['prev'] = self.hash_object(grand_object)

    def video_with_frame_drop(self, video_file, FPS=30):
        self.video = cv2.VideoCapture(video_file)
        skip_frames = 0
        t = time.time()
        try:
            while self.video.isOpened():
                for i in range(skip_frames):
                    _, _ = self.video.read()
                ret, self.image = self.video.read()
                self.result = self.tfnet.return_predict(self.image)
                self.print_box()
                cv2.waitKey(1)
                skip_frames = int((time.time()-t)*FPS)
                t = time.time()
        except AssertionError:
            pass

pred = darkflow_prediction()
# pred.image("../cars2.jpg")
pred.video("../snippet2.mp4")
pred = darkflow_prediction()
pred.video("../snippet3.mp4")
pred = darkflow_prediction()
pred.video("../snippet1.mp4")
pred = darkflow_prediction()
pred.video("../dshcm.mp4")
