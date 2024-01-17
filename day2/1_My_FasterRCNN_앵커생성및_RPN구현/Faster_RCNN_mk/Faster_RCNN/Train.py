from Faster_RCNN.Vgg16Model import Vgg16Cls
from Faster_RCNN.RPN import RPNCls
# from Faster_RCNN.RCNN import RCNNCls
from keras.models import load_model
# from Faster_RCNN.ImagePreprocess import *
# from Faster_RCNN.Anchors_IOU import AnchorsCls
import cv2
import matplotlib.pyplot as plt

class TrainCls():
    def __init__(self, NumberOfanchorboxes,  # 9
                 anchorbox_scales,  # [1, 2, 3]
                 anchorbox_ratios,  #  [0.5, 1, 2]
                 anchorbox_base_size,  # 132
                 roipool_size=(7,7), epoches = 1):
        self.anchor_k = NumberOfanchorboxes  # 매개변수를 Trainclass 멤버변수 화
        self.anchor_scales = anchorbox_scales
        self.anchor_ratios = anchorbox_ratios
        self.anchor_base_size = anchorbox_base_size
        self.classes = {'car':1}  # rcnn 모듈의 출력층 개체 클래스 구별 위한 데이터
        self.roipool_size = roipool_size
        self.epoches = epoches
        self.PreVgg16 = Vgg16Cls()  # vgg16 모델 전이학습 (17layer)활용
        self.rpn_model_dir = './Trained_Models/rpn_model.h5' # rpn모델 저장 경로
        self.rcnn_model_dir = './Trained_Models/rcnn_model.h5' # rcnn모델 저장 경로

        # Vgg16 17layer의 block5_conv3 layer outout shape에 channels 획득

        vgg16_featuremap_ChannelNum = self.PreVgg16.get_featuremap_ChannelNum()
        print('vgg16_featuremap_ChannelNum : ', vgg16_featuremap_ChannelNum) # 512


        try:
            self.rpn_model = load_model(self.rpn_model_dir)
            print('existing rpn model loading!!')
        except:
            print('new rpn model create!!')
            # anchor_k : 9 , vgg16_featuremap_ChannelNum : 512  ==>전달
            self.rpn_model = RPNCls().create_model(self.anchor_k, vgg16_featuremap_ChannelNum)
