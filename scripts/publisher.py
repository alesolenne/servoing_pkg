#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool

def talker():

    pub = rospy.Publisher("/servo_start_task", Bool, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.01) # 10hz
    a = Bool()
    a.data = True

    while not rospy.is_shutdown():

        pub.publish(a)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
