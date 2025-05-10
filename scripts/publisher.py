#!/usr/bin/env python3

import rospy
from servoing_pkg.msg import servo_init

def talker():

    pub = rospy.Publisher("/servo_start_task", servo_init, queue_size=10)

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    a = servo_init()
    a.data = True
    a.object_name = "object"

    while not rospy.is_shutdown():

        pub.publish(a)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
