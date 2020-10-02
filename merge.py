# USAGE
# python faster_facial_landmarks.py --shape-predictor shape_predictor_5_face_landmarks.dat

# import the necessary packages
from imutils import face_utils
import dlib
from multiprocessing import Process, Queue
import cv2
import numpy as np
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
import imutils
from gaze_tracking import GazeTracking
from face_function import get_face, eye_aspect_ratio
from config import config


def run():
	# Load the parameters
	conf = config()
	
	# initialize dlib's face detector (HOG-based) and then create the
	# facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(conf.shape_predictor_path)
	
	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	
	# initialize the video stream and sleep for a bit, allowing the
	# camera sensor to warm up
	# cap = cv2.VideoCapture(conf.vedio_path)
	cap = cv2.VideoCapture(0)
	if conf.vedio_path == 0:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	_, sample_frame = cap.read()
	# sample_frame = imutils.rotate(sample_frame, 90)
	
	# Introduce mark_detector to detect landmarks.
	mark_detector = MarkDetector()
	
	# Setup process and queues for multiprocessing.
	img_queue = Queue()
	box_queue = Queue()
	img_queue.put(sample_frame)
	box_process = Process(target=get_face, args=(
		mark_detector, img_queue, box_queue,))
	box_process.start()
	
	# Introduce pose estimator to solve pose. Get one frame to setup the
	# estimator according to the image size.
	height, width = sample_frame.shape[:2]
	pose_estimator = PoseEstimator(img_size=(height, width))
	
	# Introduce scalar stabilizers for pose.
	pose_stabilizers = [Stabilizer(
		state_num=2,
		measure_num=1,
		cov_process=0.1,
		cov_measure=0.1) for _ in range(6)]
	
	tm = cv2.TickMeter()
	# Gaze tracking
	gaze = GazeTracking()
	
	# loop over the frames from the video stream
	temp_steady_pose = 0
	while True:
		# grab the frame from the threaded video stream, resize it to
		# have a maximum width of 400 pixels, and convert it to
		# grayscale
		frame_got, frame = cap.read()
		
		# Empty frame
		frame_empty = np.zeros(frame.shape)
		
		# frame = imutils.rotate(frame, 90)
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		
		# check to see if a face was detected, and if so, draw the total
		# number of faces on the frame
		if len(rects) > 0:
			text = "{} face(s) found".format(len(rects))
			# cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
			# 			0.5, (0, 0, 255), 2)
			
			# Empty frame
			cv2.putText(frame_empty, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, (0, 0, 255), 2)
	
		# initialize the frame counters and the total number of blinks
		TOTAL = 0
		COUNTER = 0
		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			
			# ********************************
			# Blink detection
			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			
			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
			# Frame empty
			cv2.drawContours(frame_empty, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame_empty, [rightEyeHull], -1, (0, 255, 0), 1)
			
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < conf.EYE_AR_THRESH:
				COUNTER += 1
			
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= conf.EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				
				# reset the eye frame counter
				COUNTER = 0
			
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			# cv2.putText(frame, "Blinks: {}".format(TOTAL), (30, 60),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 90),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			
			# Frame empty
			cv2.putText(frame_empty, "Blinks: {}".format(TOTAL), (30, 60),
						cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
			cv2.putText(frame_empty, "EAR: {:.2f}".format(ear), (30, 90),
						cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
			# ********************************
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#
			# show the face number
			# cv2.putText(frame, "Face #{}".format(i + 1), (30, 120),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			
			# Frame empty
			cv2.rectangle(frame_empty, (x, y), (x + w, y + h), (0, 255, 0), 2)
			
			# show the face number
			cv2.putText(frame_empty, "Face #{}".format(i + 1), (30, 120),
						cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
			
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (x, y) in shape:
				# cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
				cv2.circle(frame_empty, (x, y), 1, (0, 255, 255), -1)
				
				
		# **********************************************************
		if frame_got is False:
			break
		
		# If frame comes from webcam, flip it so it looks like a mirror.
		if conf.vedio_path == 0:
			frame = cv2.flip(frame, 2)
		
		# Pose estimation by 3 steps:
		# 1. detect face;
		# 2. detect landmarks;
		# 3. estimate pose
		
		# Feed frame to image queue.
		img_queue.put(frame)
		
		# Get face from box queue.
		facebox = box_queue.get()
		
		if facebox is not None:
			# Detect landmarks from image of 128x128.
			face_img = frame[facebox[1]: facebox[3],
					   facebox[0]: facebox[2]]
			face_img = cv2.resize(face_img, (conf.CNN_INPUT_SIZE, conf.CNN_INPUT_SIZE))
			face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
			
			tm.start()
			marks = mark_detector.detect_marks([face_img])
			tm.stop()
			
			# Convert the marks locations from local CNN to global image.
			marks *= (facebox[2] - facebox[0])
			marks[:, 0] += facebox[0]
			marks[:, 1] += facebox[1]
			
			# Uncomment following line to show raw marks.
			# mark_detector.draw_marks(
			#     frame, marks, color=(0, 255, 0))
			
			# Uncomment following line to show facebox.
			# mark_detector.draw_box(frame, [facebox])
			
			# Try pose estimation with 68 points.
			pose = pose_estimator.solve_pose_by_68_points(marks)
			
			# Stabilize the pose.
			steady_pose = []
			pose_np = np.array(pose).flatten()
			for value, ps_stb in zip(pose_np, pose_stabilizers):
				ps_stb.update([value])
				steady_pose.append(ps_stb.state[0])
			steady_pose = np.reshape(steady_pose, (-1, 3))
			
			# Uncomment following line to draw pose annotation on frame.
			# pose_estimator.draw_annotation_box(
			# 	frame, pose[0], pose[1], color=(255, 128, 128))
			
			# Uncomment following line to draw stabile pose annotation on frame.
			# pose_estimator.draw_annotation_box(frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))
	
			# Uncomment following line to draw head axes on frame.
			# pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])
			pose_estimator.draw_axes(frame_empty, steady_pose[0], steady_pose[1])
			print('steady pose vector: {}'.format(steady_pose[0], steady_pose[1]))
		else:
			# cv2.putText(frame, "Signal loss", (200, 200),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.putText(frame_empty, "Signal loss", (200, 200),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# ******************************************************************
		# We send this frame to GazeTracking to analyze it
		gaze.refresh(frame)
		
		frame = gaze.annotated_frame()
		text = ""
		
		if gaze.is_blinking():
			text = "Blinking"
		elif gaze.is_right():
			text = "Looking right"
		elif gaze.is_left():
			text = "Looking left"
		elif gaze.is_center():
			text = "Looking center"
		
		cv2.putText(frame_empty, text, (250, 250), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
		
		left_pupil = gaze.pupil_left_coords()
		right_pupil = gaze.pupil_right_coords()
		cv2.putText(frame_empty, "Left pupil:  " + str(left_pupil), (250, 280), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
		cv2.putText(frame_empty, "Right pupil: " + str(right_pupil), (250, 310), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
		
		# ********************************************************************
		# show the frame
		# cv2.imshow("Frame", frame)
		cv2.imshow("Frame", frame_empty)
		key = cv2.waitKey(1) & 0xFF
		
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	
	# do a bit of cleanup
	cv2.destroyAllWindows()
	cap.stop()
	
if __name__ == '__main__':
	run()