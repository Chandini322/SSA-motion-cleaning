import math
from datetime import datetime
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from itertools import chain
import itertools as it
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import mediapipe as mp
import numpy as np
import cv2
import pickle
import pandas as pd
from scipy.spatial import distance


BLEND_BONES_NAMES=['SPINE_CENTER','HIP_CENTER_POSE','HIP_CENTER','SHOULDER_CENTER','EYE_CENTER','LEFT_ANKLE',
                'LEFT_ELBOW','LEFT_EYE','LEFT_FOOT_INDEX',
                'LEFT_HEEL','LEFT_HIP','LEFT_INDEX','LEFT_KNEE','LEFT_PINKY', 
                'LEFT_SHOULDER','LEFT_THUMB','LEFT_WRIST','NOSE','RIGHT_ANKLE','RIGHT_ELBOW',
                'RIGHT_EYE','RIGHT_FOOT_INDEX','RIGHT_HEEL','RIGHT_HIP',
                'RIGHT_INDEX','RIGHT_KNEE','RIGHT_PINKY','RIGHT_SHOULDER',
                'RIGHT_THUMB','RIGHT_WRIST']
MEDIAPIPE_HAND_KEYPOINTS = ["wrist", "thumb1", "thumb2", "thumb3", "thumb4",
                            "index1", "index2", "index3", "index4",
                            "middle1", "middle2", "middle3", "middle4",
                            "ring1", "ring2", "ring3", "ring4",
                            "pinky1", "pinky2", "pinky3", "pinky4"] 
ADDITIONAL_JOINTS_NAMES=['LEFT_HAND_CENTER','RIGHT_HAND_CENTER']

LEFT_HAND_KEYPOINTS =['left_' + s for s in MEDIAPIPE_HAND_KEYPOINTS] 
RIGHT_HAND_KEYPOINTS=['right_' + s for s in MEDIAPIPE_HAND_KEYPOINTS]

OUT_BLEND_KEYPOINTS=BLEND_BONES_NAMES+LEFT_HAND_KEYPOINTS+RIGHT_HAND_KEYPOINTS




motion_word_left_arm=[3,14,6,16]
motion_word_right_arm=[3,27,19,29]
motion_word_left_leg=[2,10,12,5]
motion_word_right_leg=[2,23,25,16]


motion_word_left_sub_leg=[5,9,8]
motion_word_right_sub_leg=[16,22,21]


class Motion_prediction:
    def __init__(self,pickle_file_path,n_neighbor,output_file_name) :
        self.pickle_file_path=pickle_file_path
        self.output_file_name=output_file_name
        self.n_neighbor=n_neighbor
    def read_pickle_file(self):
        with open(self.pickle_file_path, 'rb') as file:
            loaded_dict_list = pickle.load(file)
            self.whole_data = [list(dic.values()) for dic in loaded_dict_list]
        return 

    def mot_data_to_texture(self,csvdata,motion_word):
        motion_texture=[]
        for fi in (csvdata):
            motion_texture.append(list(it.chain(*[fi[i] for i in motion_word])))
        return(motion_texture)

    def find_knn(self,creating_modelmotion_texture,motion_texture,csvdata,motion_word):
        for fi in range (len(motion_texture)):    
            result_motion_texture=creating_modelmotion_texture.kneighbors([motion_texture[fi]])
            comp=list(result_motion_texture[0][0])
            indexes=list(result_motion_texture[1][0])
            gret_90=[]
            mid_70_80=[]
            gret_30f=[]
            less_30f=[]
            intersection_condition =[]
            for cop in range (len(comp)):
                if (1-comp[cop])>=0.9:
                    gret_90.append(indexes[cop])
                if (1-comp[cop]>=0.7 and 1-comp[cop]<=0.9):
                    mid_70_80.append(indexes[cop])
                if indexes[cop]>=indexes[0]+30:
                    gret_30f.append(indexes[cop])
                if indexes[cop]<=indexes[0]-30:
                    less_30f.append(indexes[cop])
                if ((indexes[cop]>=indexes[0]+30 or indexes[cop]<=indexes[0]-30) and (1-comp[cop])>0.94 and (1-comp[cop])<0.98):
                    intersection_condition.append(indexes[cop])
            knn_frame_list_avg_texture=result_motion_texture[1][0]
            if len(intersection_condition)!=0:
                filterd_knn_frame_list_avg_texture=([csvdata[i] for i in np.array(intersection_condition)])
                sum_rawdata_knn = [[0]*3 for i in range(len(motion_word))]
                for filtred_val in filterd_knn_frame_list_avg_texture:
                    currentframe_roi_keypoint=([filtred_val[i] for i in motion_word])
                    for cr in range(len(currentframe_roi_keypoint)):
                        sum_rawdata_knn[cr]=list(np.add(sum_rawdata_knn[cr], currentframe_roi_keypoint[cr]))
                mean_avg=np.divide(sum_rawdata_knn,len(np.array(intersection_condition)))
                sha=(mean_avg.shape)
                orginal_=np.reshape(motion_texture[fi],(mean_avg.shape))
                mdm_right_arm_sim=cosine_similarity(orginal_,mean_avg)
                print(mdm_right_arm_sim)
                for mo in range(len(mdm_right_arm_sim)):
                    if(mdm_right_arm_sim[mo][mo]<=0.95):
                        adjustedValues=mean_avg
                        old=csvdata[fi][motion_word[mo]]
                        csvdata[fi][motion_word[mo]]=list(adjustedValues[mo])
                        if (old==csvdata[fi][motion_word[mo]]):
                            print("chached and changed but same")
                        else:
                            print("not same but changed")
        return csvdata

    def creating_model(self,n_neighbor,motion_texture):
        creating_modelmotion_texture= NearestNeighbors(n_neighbors=n_neighbor,
                        metric='cosine',
                        algorithm='brute',
                        n_jobs=-1)
        creating_modelmotion_texture.fit(motion_texture)
        return creating_modelmotion_texture

    def motion_prediction_correction(self):
        self.read_pickle_file()
        self.motion_texture_left_arm=self.mot_data_to_texture(self.whole_data,motion_word_left_arm)
        self.motion_texture_right_arm=self.mot_data_to_texture(self.whole_data,motion_word_right_arm)
        self.motion_texture_left_leg=self.mot_data_to_texture(self.whole_data,motion_word_left_leg)
        self.motion_texture_right_leg=self.mot_data_to_texture(self.whole_data,motion_word_right_leg)
        self.motion_texture_left_leg_sub=self.mot_data_to_texture(self.whole_data,motion_word_left_sub_leg)
        self.motion_texture_right_leg_sub=self.mot_data_to_texture(self.whole_data,motion_word_right_sub_leg)

        creating_modelmotion_texture_left_arm=self.creating_model(self.n_neighbor,self.motion_texture_left_arm)
        creating_modelmotion_texture_right_arm=self.creating_model(self.n_neighbor,self.motion_texture_right_arm)
        creating_modelmotion_texture_left_leg=self.creating_model(self.n_neighbor,self.motion_texture_left_leg)
        creating_modelmotion_texture_right_leg=self.creating_model(self.n_neighbor,self.motion_texture_right_leg)
        creating_modelmotion_texture_left_leg_sub=self.creating_model(self.n_neighbor,self.motion_texture_left_leg_sub)
        creating_modelmotion_texture_right_leg_sub=self.creating_model(self.n_neighbor,self.motion_texture_right_leg_sub)

        self.find_knn(creating_modelmotion_texture_left_arm,self.motion_texture_left_arm,self.whole_data,motion_word_left_arm)
        print("left arm is done")
        self.find_knn(creating_modelmotion_texture_right_arm,self.motion_texture_right_arm,self.whole_data,motion_word_right_arm)
        print("right_arm_done")
        self.find_knn(creating_modelmotion_texture_left_leg,self.motion_texture_left_leg,self.whole_data,motion_word_left_leg)
        print("left leg is done")
        self.find_knn(creating_modelmotion_texture_right_leg,self.motion_texture_right_leg,self.whole_data,motion_word_right_leg)
        print("right_leg_done")
        self.find_knn(creating_modelmotion_texture_left_leg_sub,self.motion_texture_left_leg_sub,self.whole_data,motion_word_left_sub_leg)
        print("left leg sub is done")
        self.find_knn(creating_modelmotion_texture_right_leg_sub,self.motion_texture_right_leg_sub,self.whole_data,motion_word_right_sub_leg)
        print("right_leg_sub_done")
        return
    def update(self):
        self.motion_prediction_correction()
        self.new_csv_data=[]
        for frame_number in range (len(self.whole_data)):
            print(frame_number)
            mark={}
            for keypoint_num in range (len(self.whole_data[frame_number])):
                    mark.update({OUT_BLEND_KEYPOINTS[keypoint_num]:self.whole_data[frame_number][keypoint_num]})
            self.new_csv_data.append(mark)
        now=datetime.now()
        dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
        with open(str(self.output_file_name)+str(dt_string)+'motion_prediction.pickle', 'wb') as file:
            pickle.dump(self.new_csv_data, file)
        # df = pd.DataFrame(self.csv_data)
        # return df.to_csv(str(self.output_csv_path)+'.csv')
        return

if __name__ == '__main__':
    pickle_file_path="/home/kalpit/work/chandini/Mocap/Allah Maaf Kare_with_hands13-02-2023 21:11:09video_processor.pickle"
    output_file_name="Allah Maaf Kare_with_hands"
    n_neighbor=10
    video_process=Motion_prediction(pickle_file_path,n_neighbor,output_file_name)
    video_process.update()