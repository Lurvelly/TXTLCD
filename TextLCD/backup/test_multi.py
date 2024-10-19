import os, os.path
import re
import sys
import tarfile
import copy
import sys
import math
import rospy


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import textwrap
import numpy as np
from six.moves import urllib
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


from text_recognition import TextRecognition
from text_detection import TextDetection

from util import *
from shapely.geometry import Polygon, MultiPoint
from shapely.geometry.polygon import orient
from skimage import draw

from sensor_msgs.msg import Image as ImageMy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ros_numpy.image import numpy_to_image,image_to_numpy
from numpy import frombuffer, uint8
from cv_bridge import CvBridge
from queue import Queue
import threading


UPLOAD_FOLDER = '/ocr/ocr/uploads'
IMAGE_FOLDER = '/ocr/ocr/image/result/output'
VIDEO_FOLDER = r'/ocr/ocr/video'
FOND_PATH = '/ocr/ocr/STXINWEI.TTF'
SOURCE_FOLDER = '/ocr/ocr/image/result/txt'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4','avi'])
VIDEO_EXTENSIONS = set(['mp4', 'avi'])

import time

def init_ocr_model():
    detection_pb = './checkpoint/ICDAR_0.7.pb' # './checkpoint/ICDAR_0.7.pb'
    # recognition_checkpoint='/data/zhangjinjin/icdar2019/LSVT/full/recognition/checkpoint_3x_single_gpu/OCR-443861'
    # recognition_pb = './checkpoint/text_recognition_5435.pb' # 
    recognition_pb = './checkpoint/text_recognition.pb'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    with tf.device('/gpu:0'):
        tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),#, visible_device_list="9"),
                                   allow_soft_placement=True)

        detection_model = TextDetection(detection_pb, tf_config, max_size=1600)
        recognition_model = TextRecognition(recognition_pb, seq_len=27, config=tf_config)
    label_dict = np.load('./reverse_label_dict_with_rects.npy', allow_pickle=True)[()] # reverse_label_dict_with_rects.npy  reverse_label_dict
    return detection_model, recognition_model, label_dict 

def order_points(pts):
    def centeroidpython(pts):
        x, y = zip(*pts)
        l = len(x)
        return sum(x) / l, sum(y) / l

    centroid_x, centroid_y = centeroidpython(pts)
    pts_sorted = sorted(pts, key=lambda x: math.atan2((x[1] - centroid_y), (x[0] - centroid_x)))
    return pts_sorted

def draw_annotation_modify(image, points, label, position, horizon=True, vis_color=(30,255,255)):#(30,255,255)
    # print(type(points)) #list
    points = np.asarray(points)
    # print(type(points)) #numpy
    points = np.reshape(points, [-1, 2])
    cv2.polylines(image, np.int32([points]), 1, (0, 255, 0), 2)
    # print(type(image)) #numpy.ndarray
    # print(points)

    image = Image.fromarray(image)
    # print(type(image)) #PIL.Image.Image
    width, height = image.size
    fond_size = int(max(height, width)*0.03)
    FONT = ImageFont.truetype(FOND_PATH, fond_size, encoding='utf-8')
    DRAW = ImageDraw.Draw(image)

    original_points = points
    points = order_points(points)
    print(type(points[0]))
    print(points)
    if horizon:
        DRAW.text((points[0][0], max(points[0][1] - fond_size, 0)), label, vis_color, font=FONT) #u, v
    else:
        lines = textwrap.wrap(label, width=1)
        y_text = points[0][1]
        for line in lines:
            width, height = FONT.getsize(line)
            DRAW.text((max(points[0][0] - fond_size, 0), y_text), line, vis_color, font=FONT)
            y_text += height
    image = np.array(image)

    # sorted_original_points = sorted(original_points, key=lambda x:x[0])
    # left_points = sorted_original_points[:2]
    # center = tuple(np.int32([(left_points[0][0] + left_points[1][0]) * 0.5, (left_points[0][1] + left_points[1][1]) * 0.5]))
    mean_array = np.mean(points, axis=0)
    left_points = [point for point in points if point[0] < mean_array[0]]
    left_points = sorted(left_points, key=lambda x:x[1])
    max_distance = 0.0
    center = np.array([0, 0])# (0, 0)error, center is not correctly found
    for i in range(len(left_points) - 1):
        distance = np.linalg.norm(left_points[i] - left_points[i+1])
        if distance > max_distance:
            max_distance = distance
            center = np.int32(0.5*(left_points[i] + left_points[i+1]))
    cv2.circle(image, tuple(center), 1, (0, 0, 255), 2)
    # cv2.circle(image, tuple(np.int32([100, 200])), 1, (0, 0, 255), 2)

    right_points = [point for point in points if point[0] > mean_array[0]]
    right_points = sorted(right_points, key=lambda x:x[1])
    max_distance = 0.0
    second_center = np.array([0, 0])# (0, 0)error, center is not correctly found
    for i in range(len(right_points) - 1):
        distance = np.linalg.norm(right_points[i] - right_points[i+1])
        if distance > max_distance:
            max_distance = distance
            second_center = np.int32(0.5*(right_points[i] + right_points[i+1]))
    cv2.circle(image, tuple(second_center), 1, (255, 0, 0), 2)

    position.append(center[0])
    position.append(center[1])
    position.append(second_center[0])
    position.append(second_center[1])
    for point in points:
        position.append(point[0])
        position.append(point[1])
    return image

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def mask_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    mask = poly2mask(vertex_row_coords, vertex_col_coords, (h, w))  # y, x
    mask = np.float32(mask)
    mask = np.expand_dims(mask, axis=-1)
    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return mask, bbox

def detection(img_path, detection_model, recognition_model, label_dict, it_is_video=False):
    # if it_is_video:
    #     bgr_image = img_path
    # else:
    #     bgr_image = cv2.imread(img_path)
    bgr_image = img_path
    # bgr_image = bgr_image[0:280, :]
    print(bgr_image.shape)
    vis_image = copy.deepcopy(bgr_image)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    total_time = 0.0
    start_time = time.time()
    r_boxes, polygons, scores = detection_model.predict(bgr_image)
    total_time += time.time() - start_time
    record = ""

    for r_box, polygon, score in zip(r_boxes, polygons, scores):
        mask, bbox = mask_with_points(polygon, vis_image.shape[0], vis_image.shape[1])
        masked_image = rgb_image * mask
        masked_image = np.uint8(masked_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        height, width = cropped_image.shape[:2]
        test_size = 299
        if height >= width:
            scale = test_size / height
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            left_bordersize = (test_size - resized_image.shape[1]) // 2
            right_bordersize = test_size - resized_image.shape[1] - left_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                              right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.
        else:
            scale = test_size / width
            resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
            print(resized_image.shape)
            top_bordersize = (test_size - resized_image.shape[0]) // 2
            bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
            image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                              right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            image_padded = np.float32(image_padded) / 255.

        image_padded = np.expand_dims(image_padded, 0)
        print(image_padded.shape)

        start_time = time.time()
        results, probs = recognition_model.predict(image_padded, label_dict, EOS='EOS')
        total_time += time.time() - start_time
        print(''.join(str(result) for result in results if isinstance(result, str)))
        record += (''.join(str(result) for result in results if isinstance(result, str)))
        record += '\n'
        # print("\n".join(map(str, results)))
        print(probs)
        record += (' '.join(map(str, probs)))
        record += '\n'

        ccw_polygon = orient(Polygon(polygon.tolist()).simplify(10, preserve_topology=True), sign=1.0) #counterclockwise is equal to clockwise in uv. 10 previous is 5, it is the tolerance of simplification, smaller value means smaller change 
        pts = list(ccw_polygon.exterior.coords)[:-1]
        
        positions = []
        vis_image = draw_annotation_modify(vis_image, pts, ''.join(results), position=positions) #resort again here
        record += ' '.join(map(str, positions))
        # print(record)
        record += '\n'

        # if height >= width:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results), False)
        # else:
        #     vis_image = draw_annotation(vis_image, pts, ''.join(results))
    print("################################Time for one pic: %.6f seconds"%total_time)
    return vis_image, record

ocr_detection_model, ocr_recognition_model, ocr_label_dict = init_ocr_model()
# img_path="/ocr/ocr/data/4.jpg"
# save_path="/ocr/ocr/data/4.jpg"

class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)
        # self.image_queue = Queue(maxsize=1)

        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1)
        # self.image_sub = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.image_callback)

        # Publish to a new topic
        self.image_pub = rospy.Publisher('/usb_cam/image_processed', ImageMy, queue_size=1)
        self.string_pub = rospy.Publisher('/ocr_string', String, queue_size=1)
        
        # Initialize CvBridge
        # self.bridge = CvBridge()

        self.latest_image = None
        self.latest_timestamp = -1.0
        self.last_timestamp = -1.0
        self.lock = threading.Lock()
        self.processing_thread = threading.Thread(target=self.process)
        self.processing_thread.start()

    def process(self):
        rate = rospy.Rate(30) # todo
        while not rospy.is_shutdown():
            fetched = False
            with self.lock:
                # print("===")
                # print(self.latest_timestamp)
                # print(self.last_timestamp)
                # print("===")
                if self.latest_image is not None and self.latest_timestamp > 0:
                    timestamp_tmp = self.latest_timestamp
                    if timestamp_tmp > self.last_timestamp:
                      image_tmp = self.latest_image #todo make sure its larger
                      print("latest timestamp: ")
                      print(timestamp_tmp)
                      self.last_timestamp = timestamp_tmp
                      fetched = True
            if fetched:   
                start_time = time.time()
                camera_matrix = np.array([[616.625219502153, 0.0, 953.517490713636],
                          [0.0, 616.2007856249934, 498.49245496387283],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
                distortion_coefficients = np.array([-0.009043984784710288,
                  -0.00984168948459366, 
                  0.00013640375398168132, 
                  -0.0010776804739845835], dtype=np.float64)
                image_tmp = cv2.undistort(image_tmp, camera_matrix, distortion_coefficients)     
                image, record = detection(image_tmp, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
                # if(len(record) == 0):
                #     return
                self.image_pub.publish(numpy_to_image(image,encoding="bgr8"))
                record_msg = String()
                timestamp_string = str(timestamp_tmp) + '\n'
                record_msg.data = timestamp_string + record
                self.string_pub.publish(record_msg)
                end_time = time.time()
                print("total process time: ")
                print(end_time-start_time)
            else:
                rate.sleep()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            # print("cccc")
            # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # print("hhh")
            # cv_image=image_to_numpy(msg)

            # cv_image=numpify(msg)
            start = time.time()
            # print("current time: ")
            # print(msg.header.stamp.to_sec())
            compressed_image = msg.data
            # print("here")

            cv_image=frombuffer(compressed_image, dtype=uint8)

            cv_image=cv2.imdecode(cv_image, cv2.IMREAD_COLOR)

            # camera_matrix = np.array([[616.625219502153, 0.0, 953.517490713636],
            #               [0.0, 616.2007856249934, 498.49245496387283],
            #               [0.0, 0.0, 1.0]], dtype=np.float64)
            # distortion_coefficients = np.array([-0.009043984784710288,
            #    -0.00984168948459366, 
            #    0.00013640375398168132, 
            #    -0.0010776804739845835], dtype=np.float64)
            # cv_image = cv2.undistort(cv_image, camera_matrix, distortion_coefficients)


            # print("here3")
            with self.lock:
                # print(msg.header.stamp)
                self.latest_timestamp = msg.header.stamp.to_sec()
                # print("callback===")
                # print(msg.header.stamp.to_sec())
                # print(self.latest_timestamp)
                self.latest_image = cv_image
                
            # Convert the compressed image message to an OpenCV image
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # Perform your image processing here
            # For example, you can access the image data as a NumPy array
            # cv_image = np.asarray(cv_image)
            # print("here 3")

            # print("well")
            # print(cv_image.type)
            # Get image size
            # height, width, channels = cv_image.shape
            # print("here 4")
            #print(f"Received Image: Width={width}, Height={height}, Channels={channels}")
	    
            # print(height)
            # print(width)

            # image,_ = detection(cv_image, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
            # print("here 5")

            # Republish the received image
            # self.image_pub.publish(numpy_to_image(cv_image,encoding="rgb8"))
            # self.image_pub.publish(cv_image)
            # self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, encoding="passthrough"))
            # print("here end")
            end = time.time()
            # print("callback time: ")
            # print(end-start)
        except Exception as e:
            #rospy.logerr(f"Error processing image:")
            print(e)
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    image_processor = ImageProcessor()
    image_processor.run()

# image,_ = detection(img_path, ocr_detection_model, ocr_recognition_model, ocr_label_dict)
# cv2.imwrite(save_path, image)
