import rosbag, rospy, numpy as np
import sys, os, cv2, glob
from itertools import repeat

from tqdm import tqdm, trange

# try to find cv_bridge:
try:
    from cv_bridge import CvBridge
except ImportError:
    # assume we are on an older ROS version, and try loading the dummy manifest
    # to see if that fixes the import error
    try:
        import roslib; roslib.load_manifest("bag2video")
        from cv_bridge import CvBridge
    except:
        print ("Could not find ROS package: cv_bridge")
        print ("If ROS version is pre-Groovy, try putting this package in ROS_PACKAGE_PATH")
        sys.exit(1)

def get_info(bag, topic=None, start_time=rospy.Time(0), stop_time=rospy.Time(sys.maxsize)):
    size = (0,0)
    times = []

    # read the first message to get the image size
    msg = next(bag.read_messages(topics=topic))[1]
    size = (msg.width, msg.height)

    # now read the rest of the messages for the rates
    iterator = bag.read_messages(topics=topic, start_time=start_time, end_time=stop_time)#, raw=True)
    for _, msg, _ in iterator:
        time = msg.header.stamp
        times.append(time.to_sec())
        size = (msg.width, msg.height)
    diffs = 1/np.diff(times)
    return (np.median(diffs), min(diffs), max(diffs), size, times)

def calc_n_frames(times, precision=10):
    # the smallest interval should be one frame, larger intervals more
    intervals = np.diff(times)
    return (np.int64(np.round(precision*intervals/min(intervals))))

def write_frames(bag, writer, total, topic=None, nframes=repeat(1), start_time=rospy.Time(0), stop_time=rospy.Time(sys.maxsize), viz=False, encoding='bgr8'):
    bridge = CvBridge()
    if viz:
        cv2.namedWindow('win')
    count = 1
    iterator = bag.read_messages(topics=topic, start_time=start_time, end_time=stop_time)
    for (topic, msg, time), reps in zip(iterator, nframes):
        print ('Writing frame %s of %s at time %s' % (count, total, time), end='\r')
        img = np.asarray(bridge.imgmsg_to_cv2(msg, 'bgr8'))
        for rep in range(reps):
            writer.write(img)
        # imshow('win', img)
        count += 1

def imshow(win, img):
    cv2.imshow(win, img)
    cv2.waitKey(1)

def noshow(win, img):
    pass

def make_video(bag, outfile):
    

    image_topic = list(bag.get_type_and_topic_info()[1].keys())[1]

    print ('Calculating video properties')
    rate, minrate, maxrate, size, times = get_info(bag, image_topic, start_time=rospy.Time(0), stop_time=rospy.Time(sys.maxsize))
    nframes = calc_n_frames(times, 10)
    # writer = cv2.VideoWriter(outfile, cv2.cv.CV_FOURCC(*'DIVX'), rate, size)
    writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MP4V'),  np.ceil(maxrate*10), size)
    print ('Writing video')
    write_frames(bag, writer, len(times), topic=image_topic, nframes=nframes, start_time=rospy.Time(0), stop_time=rospy.Time(sys.maxsize), encoding='bgr8')
    writer.release()
