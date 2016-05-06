#!/usr/bin/env python
import os
import cv2
import sys
import rospy
import numpy as np
import cv2.cv as cv
from cv_bridge import CvBridge
from collections import Counter
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from clothes_detection.msg import Shirt
from scipy.cluster.vq import vq, kmeans
from clothes_detection.msg import MainMsg
from clothes_detection.msg import ImgArray
from clothes_detection.msg import BoundingBox

storage = None
image_topic = ""
frame_counter = 0
publish_all = False
publish_face = False
cascadeFrontal = None
image_subscriber = None
mainmsg_publisher = None
face_image_publisher = None
whole_image_publisher = None
HAAR_CASCADE_PATH_FRONTAL = ""

'''
The 3 functions below are responsible for face detection.

Provided by Dr Theodore Giannakopoulos
'''

def intersect_rectangles(r1, r2):
	x11 = r1[0]; y11 = r1[1]; x12 = r1[0]+r1[2]; y12 = r1[1]+r1[3];
	x21 = r2[0]; y21 = r2[1]; x22 = r2[0]+r2[2]; y22 = r2[1]+r2[3];
		
	X1 = max(x11, x21); X2 = min(x12, x22);
	Y1 = max(y11, y21); Y2 = min(y12, y22);

	W = X2 - X1
	H = Y2 - Y1
	if (H>0) and (W>0):
		E = W * H;
	else:
		E = 0.0;
	Eratio = 2.0*E / (r1[2]*r1[3] + r2[2]*r2[3])
	return Eratio

def initialize_face():
	global storage, cascadeFrontal, HAAR_CASCADE_PATH_FRONTAL
	try:
		cascadeFrontal = cv2.cv.Load(HAAR_CASCADE_PATH_FRONTAL);
		storage = cv2.cv.CreateMemStorage()
	except ValueError:
		print ValueError
	return (cascadeFrontal, storage)

def detect_faces(image, cascadeFrontal, storage):
	facesFrontal = []; 	

	detectedFrontal = cv2.cv.HaarDetectObjects(image, cascadeFrontal, storage, 1.3, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (image.width/10, image.width/10))
	
	if detectedFrontal:
		for (x,y,w,h),n in detectedFrontal:
			facesFrontal.append((x,y,w,h))

	# remove overlaps:
	while (1):
		Found = False
		for i in range(len(facesFrontal)):
			for j in range(len(facesFrontal)):
				if i != j:
					interRatio = intersect_rectangles(facesFrontal[i], facesFrontal[j])
					if interRatio>0.3:
						Found = True;
						del facesFrontal[i]
						break;
			if Found:
				break;

		if not Found:	# not a single overlap has been detected -> exit loop
			break;
	return (facesFrontal)
	



def imageCallback(image_message):
	#TODO add global
	global publish_face, publish_all
	global face_image_publisher, mainmsg_publisher, whole_image_publisher
	
	faces_rect = []
	shirts_rect = []
	img_array = []
	bridge = CvBridge()
	face_checker  = False
	shirt_checker = False
	shirt_color = (255,255,255)
	
	cv_image = CvBridge().imgmsg_to_cv2(image_message, "bgr8")
	
	height, width, channels = cv_image.shape
	facesFrontal = detect_faces(cv2.cv.fromarray(cv_image), cascadeFrontal, storage)
		
	for f in facesFrontal:
		#boolean values to synchronize the rectangles
		face_checker  = False
		shirt_checker = False					
		
		##################
		# FACE rectangle #
		##################
		if publish_face:			
			img_array.append(bridge.cv2_to_imgmsg(cv_image[f[1]:f[1]+f[3],f[0]:f[0]+f[2]], "bgr8"))

		cv2.rectangle(cv_image, (f[0], f[1]+7), (f[0]+f[2],f[1]+f[3]), (0,255,255), -1)
		face_roi = cv_image[f[1]:f[1]+f[3],f[0]:f[0]+f[2]]

		if publish_all:
			rect_text = "Face\nFeatures"
			y0 = 50
			dy = 20
			for i, t in enumerate(rect_text.split('\n')):
				y = y0+i*dy
				cv2.putText(cv_image,t, (f[0]+20, f[1]+y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		
		face_height, face_width, channels = face_roi.shape # finds the height and the width of the face roi
		if(face_height>0 and face_width>0):
			face_checker = True	 

			p = BoundingBox()
			p.top_left.x = f[0]
			p.top_left.y = f[1]+7
			p.bottom_right.x = f[0]+f[2]
			p.bottom_right.y = f[1]+f[3]

			faces_rect.append(p)

		###################
		# SHIRT rectangle #
		###################
		shirt_roi = cv_image[f[1]+f[3]+20+height/12:f[1]+f[3]+2+height/3, f[0]+30-height/7:f[0]+f[2]-30+height/7]
		# Finds the height and the width of the shirt roi
		shirt_height, shirt_width, channels = shirt_roi.shape
		
		'''
		# Initialize shirt dimensions, in order the program not to crash
		s_left   = 0
		s_up     = 0
		s_right  = 5
		s_down   = 5	 
	 	s_height = 0
	 	s_width  = 0
		'''
		if(shirt_height>0 and shirt_width>0): # checks if the rectangle actually appears
		   	shirt_checker=True   
		   	s_left   = f[0]+30-height/7
		   	s_right  = f[0]+f[2]-30+height/7
		   	s_up     = f[1]+f[3]+20+height/12
		   	s_down   = f[1]+f[3]+2+height/3
		   	s_height = s_up-s_down
		   	s_width  = s_left-s_right

		p = Shirt()

		if face_checker:
			p.bounding_box.top_left.x = -1
			p.bounding_box.top_left.y = -1
			p.bounding_box.bottom_right.x = -1
			p.bounding_box.bottom_right.y = -1


		c = (0, 0, 0) #init colour

		if(face_checker and shirt_checker):
			Z = shirt_roi
			Z = np.float32(Z)
			Z = np.transpose(Z)
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
			K = 2
			try:
				ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
				colour_ = []
				if list(label).count([0]) > list(label).count([1]):
					colour_ = Z[np.where(label==[0])]
				else:
					colour_ = Z[np.where(label==[1])]
				c = (int(colour_[0][0]),int(colour_[0][1]),int(colour_[0][2]))
				if publish_all:
					cv2.rectangle(cv_image, (f[0]+30-height/7, f[1]+f[3]+30+height/12), (f[0]+f[2]-10+height/7,f[1]+f[3]+2+height/3), c, -1)
					rect_text = "Shirt Colour"
					cv2.putText(cv_image,rect_text, (f[0]+40-height/7, f[1]+f[3]+y+height/12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255))

				p.bounding_box.top_left.x = f[0]+30-height/7
				p.bounding_box.top_left.y = f[1]+f[3]+30+height/12
				p.bounding_box.bottom_right.x = f[0]+f[2]-10+height/7
				p.bounding_box.bottom_right.y = f[1]+f[3]+2+height/3
				#bgr8
				p.red = c[2]
				p.green = c[1]
				p.blue = c[0]

			except:
				print sys.exc_info()[0]

		if face_checker:
			shirts_rect.append(p)

	h = Header()
	h.stamp = rospy.get_rostime()

	if publish_all:
		whole_image_publisher.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))

	if publish_face:
		m = ImgArray()
		m.header = h
		m.images = img_array
		face_image_publisher.publish(m)
	
	if face_checker:
		msg = MainMsg()
		msg.header = h
		msg.faces = faces_rect
		msg.shirts = shirts_rect
		mainmsg_publisher.publish(msg)


def initNode():
	global image_subscriber, image_topic, publish_face, publish_all, HAAR_CASCADE_PATH_FRONTAL
	global face_image_publisher, mainmsg_publisher, whole_image_publisher

	rospy.init_node("clothes_detection")

	publish_face = rospy.get_param('/clothes_detection/publish_face', False)
	publish_all = rospy.get_param('/clothes_detection/publish_whole_image', False)
	image_topic = rospy.get_param('/clothes_detection/image_topic', '/usb_cam/image_raw')
	main_message_topic = rospy.get_param('/clothes_detection/main_message_topic', '/clothes_detection/face_rectangle')
	whole_image_message_topic = rospy.get_param('/clothes_detection/whole_image_message_topic', '/clothes_detection/image')
	face_image_message_topic = rospy.get_param('/clothes_detection/face_image_message_topic', '/clothes_detection/face_image')

	mainmsg_publisher = rospy.Publisher(main_message_topic, MainMsg, queue_size=1)
	#shirt_rect_publisher = rospy.Publisher(shirt_message_topic, Shirts, queue_size=1)
	whole_image_publisher = rospy.Publisher(whole_image_message_topic, Image, queue_size=10)
	face_image_publisher = rospy.Publisher(face_image_message_topic, ImgArray, queue_size=10)

	HAAR_CASCADE_PATH_FRONTAL = os.path.dirname(os.path.realpath(sys.argv[0])) + "/data/haarcascades/haarcascade_frontalface_default.xml" # loads the face detector cascade classifier
	
	(cascadeFrontal, storage) = initialize_face() # initialize face detector

	image_subscriber = rospy.Subscriber(image_topic, Image, imageCallback)
	while not rospy.is_shutdown():
		rospy.spin()


if __name__ == '__main__':
	initNode()