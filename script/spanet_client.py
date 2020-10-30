#!/usr/bin/env python

import rospy
import yaml

from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from food_msg.srv import SPANetTrigger, SPANetTriggerRequest

class Visualizer(object):
    "viz module, edited by ed209"
    "test"
    def __init__(self, img_topic="/rm/bbox_img"):
        self.visual_pause = 0.2
        self.img_pub = rospy.Publisher(img_topic, Image, queue_size=1)
        self.markers_pub = rospy.Publisher("food_detector/marker_array", MarkerArray, queue_size=1)

    def viz_bbox_img(self, bbox_img_msg):
        start = rospy.get_time()
        while True:
            self.img_pub.publish(bbox_img_msg)
            time_len = rospy.get_time() - start
            if time_len > self.visual_pause:
                break

    def viz_markers(self, markers):
        start = rospy.get_time()
        while True:
            self.markers_pub.publish(markers)
            time_len = rospy.get_time() - start
            if time_len > self.visual_pause:
                break

    def viz_all(self, markers, bbox_img_msg):
        self.viz_bbox_img(bbox_img_msg)
        self.viz_markers(markers)


    def img_stiching(self):
        pass

class ReconfigManager(Visualizer):

    def __init__(self, service_name='SPANet'):
        Visualizer.__init__(self, "/rm/bbox_img")
        self.thresh = 0.7
        self.service_name = service_name
        self.spanet_client = rospy.ServiceProxy(service_name, SPANetTrigger)

    def detect(self):
        rospy.wait_for_service(self.service_name, timeout=5)
        res = self.spanet_client(SPANetTriggerRequest())
        return res.markers, res.bbox_img_msg

    def find_best_marker(self, markers):
        best_score = 0
        best_idx = 0
        for i, marker in enumerate(markers):
            yaml_node = yaml.load(marker.text)
            score = yaml_node['score']
            if best_score < score:
                best_score = score
                best_idx = i
        return markers[best_idx], best_score

    def rm_spin(self):
        while not rospy.is_shutdown():
            markers, bbox_img_msg = self.detect()
            best_marker, best_score = self.find_best_marker(markers)
            self.viz_bbox_img(bbox_img_msg)
            print("the best score = {}".format(best_score))
            val = input("Press: \n1 to push, 2 to scoop, 0 to do nothing, -1 to exit\n")
            if val == -1:
                break
            if val == 0:
                pass
            if val == 1:
                pass
            if val == 2:
                self.viz_all(markers, bbox_img_msg)

if __name__ == "__main__":
    rospy.init_node("spanet_client")
    rm = ReconfigManager()
    rm.rm_spin()
