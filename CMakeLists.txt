cmake_minimum_required(VERSION 3.0.2)
project(pytorch_pix2food)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package(
#  INCLUDE_DIRS include
 LIBRARIES pytorch_pix2food
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)
