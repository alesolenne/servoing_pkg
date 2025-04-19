#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from servoing_pkg.symbolic import test
from servoing_pkg.jac import test2
from servoing_pkg.kinematics import *

def publisher_node():
    rospy.init_node('test_publisher', anonymous=True)
    pub = rospy.Publisher('test_topic', String, queue_size=10)

    while not rospy.is_shutdown():
        message = "Hello, ROS!"
        c = test(0.1,0.1,0.2,-0.1,0.1,0.5,0.2,0.4)
        a = test2(0.1,0.1,0.2,-0.1,0.1,0.5,0.2,0.4)  # Call the test function with dummy values
        print(a)
        pub.publish(message)

if __name__ == '__main__':
    try:
        publisher_node()
    except rospy.ROSInterruptException:
        pass