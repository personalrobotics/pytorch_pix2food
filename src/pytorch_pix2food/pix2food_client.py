#!/usr/bin/python
import rospy
import rospkg
import os
import cv2
from cv_bridge import CvBridge
from pytorch_pix2food.dataset.utils import generateActionImg
from food_msg.srv import Pix2Food, Pix2FoodRequest

rospack = rospkg.RosPack()
pix2food_pkg = rospack.get_path('pytorch_pix2food')
br = CvBridge()

if __name__ == "__main__":
    rospy.init_node("pix2food_client")
    # -------------- Action IMG --------------- #    
    bbox=[75, 118, 300, 162]
    xmin, ymin, xmax, ymax = bbox
    start = [xmax, (ymin + ymax) // 2]
    end = [xmin, (ymin + ymax) // 2]
    actImg = generateActionImg(start, end, actImg=None, push_direction="left_push", 
                            img_size = (640, 480), forque_width=45)
    # -------------- Start IMG --------------- #
    script_base = os.path.join(pix2food_pkg, "script")
    img_path = os.path.join(script_base, "0060_1_start.png")
    startImg = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # -------------- cv 2 msg --------------- #
    startImg = br.cv2_to_imgmsg(startImg)
    actImg = br.cv2_to_imgmsg(actImg)
    # -------------- call service --------------- #
    s = rospy.ServiceProxy("pix2food", Pix2Food)
    res = s(Pix2FoodRequest(
        startImg=startImg,
        actImg=actImg
    ))

    if res is not None:
        fakeImg = cv2.cvtColor(br.imgmsg_to_cv2(res.fakeImg), cv2.COLOR_BGR2RGB)
        cv2.imshow("fakeImg", fakeImg)
        cv2.waitKey(0)
    else:
        print("it's None")

    
