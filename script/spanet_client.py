#!/usr/bin/env python

import rospy
import yaml
import cv2
import numpy as np

from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from food_msg.srv import SPANet, SPANetRequest, Pix2Food, Pix2FoodRequest
from cv_bridge import CvBridge
from pose_estimators.utils import CameraSubscriber
from pytorch_pix2food.dataset.utils import generateActionImg

class SquirrelLogger(object):
    def __init__(self):
        print("Logger Init")

    def info(self, msg):
        print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(msg)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")                 

class Visualizer(object):
    "viz module, edited by ed209"
    "test"
    def __init__(self, img_topic="/rm/bbox_img"):
        self.br = CvBridge()
        self.visual_pause = 0.1
        self.raw_pub = rospy.Publisher("/nansong/original", Image, queue_size=1)
        self.pushed_pub = rospy.Publisher("/nansong/pushed", Image, queue_size=1)
        self.markers_pub = rospy.Publisher("/food_detector/marker_array", MarkerArray, queue_size=1)
        self.rate = rospy.Rate(20)
    def viz_bbox_img(self, bbox_img_msg, pushed=False):
        publisher = self.pushed_pub if pushed else self.raw_pub
        start = rospy.get_time()
        while True:
            publisher.publish(bbox_img_msg)
            time_len = rospy.get_time() - start
            self.rate.sleep()
            if time_len > self.visual_pause:
                break

    def viz_markers(self, markers):
        start = rospy.get_time()
        while True:
            self.markers_pub.publish(markers)
            time_len = rospy.get_time() - start
            self.rate.sleep()
            if time_len > self.visual_pause:
                break

    def viz_all(self, markers, bbox_img_msg, pushed=False):
        self.viz_bbox_img(bbox_img_msg, pushed)
        print("publishing \n", markers)
        self.viz_markers(markers)

    def img_stiching(self, bbox_img_msg, actImg):
        bbox_img = self.br.imgmsg_to_cv2(bbox_img_msg)
        h, w = actImg.shape
        actImg_CH3 = np.zeros((h, w, 3), dtype=np.uint8)
        actImg_CH3[:,:,0] = actImg_CH3[:,:,1] = actImg_CH3[:,:,2] = actImg
        stiched_img = cv2.hconcat([bbox_img, actImg_CH3])
        return self.br.cv2_to_imgmsg(stiched_img, 'rgb8')

class ReconfigManager(CameraSubscriber, Visualizer):

    def __init__(self):
        image_topic='/camera/color/image_raw/compressed'
        image_msg_type='compressed'
        depth_image_topic='/camera/aligned_depth_to_color/image_raw'
        camera_info_topic='/camera/color/camera_info'
        CameraSubscriber.__init__(
        self,
        image_topic=image_topic,
        image_msg_type=image_msg_type,
        depth_image_topic=depth_image_topic,
        pointcloud_topic=None,
        camera_info_topic=camera_info_topic)
        Visualizer.__init__(self, "/rm/bbox_img")
        self.pix2food_client = rospy.ServiceProxy('pix2food', Pix2Food)
        self.spanet_client = rospy.ServiceProxy('SPANet', SPANet)
        self.sl = SquirrelLogger()
        self.init_param()
    
    def init_param(self):
        self.thresh = 0.7
        self.best_score = 0

    def detect(self, raw_img=None):
        rospy.wait_for_service('SPANet', timeout=5)
        raw_img_msg = self.br.cv2_to_imgmsg(raw_img)
        rqt = SPANetRequest(
            raw_img_msg=raw_img_msg
        )
        res = self.spanet_client(rqt)
        return res.markers, res.bbox_img_msg

    def find_best_marker(self, markers):
        if len(markers) == 0:
            return None, 0
        best_score = 0
        best_idx = 0
        for i, marker in enumerate(markers):
            yaml_node = yaml.load(marker.text)
            score = yaml_node['score']
            if best_score < score:
                best_score = score
                best_idx = i
        return markers[best_idx], best_score

    def add_push_info(self, marker):
        yaml_node = yaml.load(marker.text)
        yaml_node["push_direction"] = "left_push"
        marker.text = yaml.dump(yaml_node)
        return marker

    def generateActImg(self, bbox=None):
        if bbox == None:
            bbox=[75, 118, 300, 162]
            # bbox=[200, 138, 360, 192]
        else:
            forque_width = 45
            push_length = 225
            xmin, ymin, xmax, ymax = bbox
            ymean = ymin + ymax
            Ymin = ymean - forque_width // 2
            Ymax = ymean + forque_width // 2 
            Xmax = xmax
            Xmin = Xmax - push_length
            bbox = [Xmin, Ymin, Xmax, Ymax]
        xmin, ymin, xmax, ymax = bbox
        start = [xmax, (ymin + ymax) // 2]
        end = [xmin, (ymin + ymax) // 2]
        actImg = generateActionImg(start, end, actImg=None, push_direction="left_push", 
                                img_size = (640, 480), forque_width=45)
        return actImg

    def generatePushedImg(self):
        img_path = "0061_5_finish.png"
        img = cv2.imread(img_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def call_pix2food(self, startImg, actImg):
        rospy.wait_for_service("pix2food", timeout=5)
        startImg, actImg = self.br.cv2_to_imgmsg(startImg), self.br.cv2_to_imgmsg(actImg)
        res = self.pix2food_client(Pix2FoodRequest(
            startImg=startImg,
            actImg=actImg
        ))
        fakeImg = self.br.imgmsg_to_cv2(res.fakeImg)
        #TODO figure out the best API input data type? Check out pix2food model
        # print(np.max(fakeImg), np.min(fakeImg))
        cv2.normalize(fakeImg, fakeImg, 0, 255, cv2.NORM_MINMAX)
        fakeImg = fakeImg.astype(np.uint8)
        return fakeImg

    def reconfig_process(self):
        actImg = self.generateActImg()
        fakeImg_left = self.call_pix2food(self.img, actImg)
        # fakeImg_left = self.generatePushedImg()
        markers, bbox_img_msg = self.detect(fakeImg_left)
        stiched_img_msg = self.img_stiching(bbox_img_msg, actImg)
        h_marker, h_score = self.find_best_marker(markers)
        if h_score > self.best_score:
            self.best_score = h_score
            # self.best_marker = h_marker
            print("push is better")
            self.best_marker = self.add_push_info(self.best_marker)
            self.viz_all([self.best_marker], stiched_img_msg, pushed=True)
            #TODO: Publish pushing msg for low level controller
        else:
            self.viz_bbox_img(bbox_img_msg, pushed=True)

    def rm_spin(self):
        while not rospy.is_shutdown():
            begin = input("begin to detect? No: 0, Yes: 1, Break: -1\n")
            if begin == -1: break
            if begin == 0: continue

            self.wait_img_msg()  # check self.img is not None
            markers, bbox_img_msg = self.detect(self.img)
            self.best_marker, self.best_score = self.find_best_marker(markers)
            self.viz_bbox_img(bbox_img_msg)
            if self.best_score < self.thresh:
                self.sl.info("the best score = {} < {}, need to reconfig via push".format(self.best_score, self.thresh))
            else:
                self.sl.info("the best score = {}, ready to scoop".format(self.best_score))

            val = input("Press: \n1 to push, 2 to scoop, 0 to do nothing\n")
            if val == 0:
                pass
            if val == 1:
                self.reconfig_process()
            if val == 2:
                self.viz_all([self.best_marker], bbox_img_msg)

if __name__ == "__main__":
    rospy.init_node("spanet_client")
    rm = ReconfigManager()
    rm.rm_spin()
