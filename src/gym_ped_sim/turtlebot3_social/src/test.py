import random
import numpy as np
import rospkg
import rospy
from lxml import etree
from lxml.etree import Element
from copy import deepcopy
import yaml

rospack = rospkg.RosPack()

if __name__ == '__main__':
	plugin_pkg_path = rospack.get_path("actor_plugin")
	plugin_path = plugin_pkg_path + "/lib/libactorplugin_ros.so"
	actor_pkg_path = rospack.get_path("actor_services")

	# world_name = rospy.get_param("BASE_WORLD")

	# tree_ = etree.parse(actor_pkg_path+'/worlds/'+world_name)
	print "tree_: ", plugin_path