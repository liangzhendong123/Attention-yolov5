#! /usr/bin/python3
"""
    订阅来自相机的消息，检测后发布检测到的绝缘子的坐标
"""

# 为了使ros可以识别调用的其他库
import sys
import os
sys.path.append(os.getcwd())

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge,CvBridgeError
import numpy as np

import torch
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized
from utils.plots import plot_one_box, colors
from models.experimental import attempt_load

"""
    使用Yolov5检测绝缘子串
"""
class Yolov5:
    """
        参数：
            weight_path:权重参数的路径
            half:是否使用int8类型
            image_size:模型输入大小
            conf_thres:置信度阈值
            iou_thres:nms的iou阈值
            max_detections:最大检测数目
    """
    def __init__(self, weight_path, half=False, image_size=640, conf_thres=0.1, iou_thres=0.45, max_detections=10):
        # 设置参数
        self.weight_path_ = weight_path # 权重路径
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设备类型（0表示gpu）
        self.half_ = half   # 是否采用int8类型
        self.image_size_ = image_size   # 推理图片大小
        self.conf_thres_ = conf_thres   # 置信度大小
        self.iou_thres_ = iou_thres     # nms iou阈值
        self.max_detections_ = max_detections   # 最大检测目标

        # 初始化model
        self.model_ = attempt_load(weight_path, map_location=self.device_)
        self.model_ = self.model_.eval()

        # 根据类型转换为gpu或cpu
        if self.device_.type != 'cpu':
            self.model_(torch.zeros(1, 3, self.image_size_, self.image_size_).to(self.device_).type_as(next(self.model_.parameters())))  # run once
        
    """
        输入的image类型必须是cv2的图片格式（即numpy格式）
    """
    def inference(self, image):
        image_plot = image.copy()

        # 图片预处理
        image = cv2.resize(image, dsize=(self.image_size_, self.image_size_), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image).to(self.device_)
        image = image.half() if self.half_ else image.float()
        image /= 255.0
        
        # 改变纬度 
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)
    
        # 推理
        t1 = time_synchronized()
        pred = self.model_(image, augment=False)[0] 
        print(pred.shape)
        # NMS 
        pred = non_max_suppression(pred, self.conf_thres_, self.iou_thres_, None, None, max_det=self.max_detections_)
        t2 = time_synchronized()
        # 处理结果
        gn = torch.tensor(image_plot.shape)[[1, 0, 1, 0]]
        
        for i, det in enumerate(pred):
            if len(det):
                print("found {} insulators, cost time: {}s".format(len(det), t2 - t1))

                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image_plot.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{'insulator'} {conf:.2f}"
                    plot_one_box(xyxy, image_plot, label=label, color=colors(int(cls), True), line_thickness=2)
        return image_plot
            
        
"""
    ros节点类，负责订阅发布消息，并调用Yolo来检测数据
"""
class Yolov5RosNode:
    """
        参数：
            image_sub_name:订阅的图片发布的topic名
            positon_pub_name:发布的位置信息的topic名
            image_pub_name:发布的检测后的图片的topic名
            weight_path:权重的路径
    """
    def __init__(self, image_sub_name, positon_pub_name, image_pub_name, weight_path):
        # 初始化ros节点和订阅发布消息
        rospy.init_node("yolov5", anonymous=True)

        self.image_sub_name_ = image_sub_name
        self.positon_pub_name_ = positon_pub_name
        self.image_pub_name_ = image_pub_name

        self.positon_pub_ = None
        self.image_pub_ = None

        self.bridge_ = CvBridge()

        self.yolov5_= Yolov5(weight_path, iou_thres=1, max_detections=30000)

    """
        回调函数，每次接收到图片后都会调用
    """
    def callback(self, data):
        try:
            image = self.bridge_.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as error:
            print(error)

        result = self.yolov5_.inference(image)
        self.image_pub_.publish(self.bridge_.cv2_to_imgmsg(result, "bgr8"))

    def run(self):
        # 订阅并发布结果
        self.positon_pub_ = rospy.Publisher(self.positon_pub_name_, Image, queue_size=3)
        self.image_pub_ = rospy.Publisher(self.image_pub_name_, Image, queue_size=3)
        rospy.Subscriber(self.image_sub_name_, Image, self.callback)

        rospy.spin()

def main():
    image_sub_name = "galaxy_camera/image_raw"
    positon_pub_name = "yolov5/position"
    image_pub_name = "yolov5/image_raw"
    weight_path = "weights/yolov5s.pt"

    detector = Yolov5RosNode(image_sub_name, positon_pub_name, image_pub_name, weight_path)
    detector.run()

if __name__ == "__main__":
    main()
