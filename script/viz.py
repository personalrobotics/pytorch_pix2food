import numpy as np
import cv2
import rospy

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge
br = CvBridge()

class Viz(object):

    def __init__(self):
        self.viz_reset()
        # viz topics

        self.detection_img = None
        self.spanet_img = []

        self.detection_img_topic = '/food_spanet_detector/detection_image'
        self.spanet_img_topic = '/food_spanet_detector/spanet_image'
        self.detection_pub_topic = '/ReconfigManager/detection_image'

        # viz topics

        self.detection_sub = rospy.Subscriber(self.detection_img_topic, Image,
                                               self.detection_callback, queue_size=1)
        self.spanet_img_sub = rospy.Subscriber(self.spanet_img_topic, Image,
                                               self.spanet_img_callback, queue_size=1)
        self.detection_pub = rospy.Publisher(
            self.detection_pub_topic,
            Image,
            queue_size=1)

    def detection_callback(self, img):
        img = br.imgmsg_to_cv2(img)
        # img = cv2.resize(img, (320, 240))
        self.detection_img = img

    def spanet_img_callback(self, img):
        img = br.imgmsg_to_cv2(img)
        img = cv2.resize(img, (640, 480))
        self.spanet_img.append(img)

    def viz_callback_reset(self):
        self.detection_img = None
        self.spanet_img = []

    def viz_reset(self):
        self.detection_imgs = []
        self.spanet_imgs = []    

    def viz_pub(self):
        self.viz_draw()

    def viz_draw(self, img_size = (640, 480)):
        # num = len(self.detection_imgs)
        # print("there's ", num, "detection imgs")
        # new_detection = cv2.vconcat(self.detection_imgs)
        # new_detection = br.cv2_to_imgmsg(new_detection, encoding="rgb8")
        iteration = 0
        for img_buffer in self.spanet_imgs:
            iteration = max(iteration, len(img_buffer))
        for j in range(iteration):
            viz_list = []
            for i, img in enumerate(self.detection_imgs):
                j_real = j % len(self.spanet_imgs[i])
                spanet_img = self.spanet_imgs[i][j_real]
                viz_list.append(cv2.hconcat([img, spanet_img]))
            new_detection = cv2.vconcat(viz_list)
            new_detection = br.cv2_to_imgmsg(new_detection, encoding="rgb8")
            self.detection_pub.publish(new_detection)
            rospy.sleep(0.4)

