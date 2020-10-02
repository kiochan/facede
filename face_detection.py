from imutils import face_utils
import imutils
import dlib
from multiprocessing import Process, Queue
import cv2
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer
from gaze_tracking import GazeTracking
from face_function import get_face, eye_aspect_ratio
from config import config
import numpy as np


class face_detector():
	def __init__(self):
		# Load the parameters
		self.conf = config()
		# initialize dlib's face detector (HOG-based) and then create the
		# facial landmark predictor
		print("[INFO] loading facial landmark predictor...")
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.conf.shape_predictor_path)
		
		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		
		# initialize the video stream and sleep for a bit, allowing the
		# camera sensor to warm up
		self.cap = cv2.VideoCapture(0)
		if self.conf.vedio_path == 0:
			self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		_, sample_frame = self.cap.read()
		
		# Introduce mark_detector to detect landmarks.
		self.mark_detector = MarkDetector()
		
		# Setup process and queues for multiprocessing.
		self.img_queue = Queue()
		self.box_queue = Queue()
		self.img_queue.put(sample_frame)
		self.box_process = Process(target=get_face, args=(
			self.mark_detector, self.img_queue, self.box_queue,))
		self.box_process.start()
		
		# Introduce pose estimator to solve pose. Get one frame to setup the
		# estimator according to the image size.
		self.height, self.width = sample_frame.shape[:2]
		self.pose_estimator = PoseEstimator(img_size=(self.height, self.width))
		
		# Introduce scalar stabilizers for pose.
		self.pose_stabilizers = [Stabilizer(
			state_num=2,
			measure_num=1,
			cov_process=0.1,
			cov_measure=0.1) for _ in range(6)]
		
		self.tm = cv2.TickMeter()
		# Gaze tracking
		self.gaze = GazeTracking()
	
	def detect(self):
		# loop over the frames from the video stream
		temp_steady_pose = 0
		while True:
			# grab the frame from the threaded video stream, resize it to
			# have a maximum width of 400 pixels, and convert it to
			# grayscale
			frame_got, frame = self.cap.read()
			
			# Empty frame
			frame_empty = np.zeros(frame.shape)
			
			# frame = imutils.rotate(frame, 90)
			frame = imutils.resize(frame, width=400)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			# detect faces in the grayscale frame
			rects = self.detector(gray, 0)
			
			# initialize the frame counters and the total number of blinks
			TOTAL = 0
			COUNTER = 0
			# loop over the face detections
			for (i, rect) in enumerate(rects):
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				self.shape = self.predictor(gray, rect)
				self.shape = face_utils.shape_to_np(self.shape)
				
				# ********************************
				# Blink detection
				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				self.leftEye = self.shape[self.lStart:self.lEnd]
				self.rightEye = self.shape[self.rStart:self.rEnd]
				self.leftEAR = eye_aspect_ratio(self.leftEye)
				self.rightEAR = eye_aspect_ratio(self.rightEye)
				
				# average the eye aspect ratio together for both eyes
				ear = (self.leftEAR + self.rightEAR) / 2.0
				
				# check to see if the eye aspect ratio is below the blink
				# threshold, and if so, increment the blink frame counter
				if ear < self.conf.EYE_AR_THRESH:
					COUNTER += 1
				
				# otherwise, the eye aspect ratio is not below the blink
				# threshold
				else:
					# if the eyes were closed for a sufficient number of
					# then increment the total number of blinks
					if COUNTER >= self.conf.EYE_AR_CONSEC_FRAMES:
						TOTAL += 1
					
					# reset the eye frame counter
					COUNTER = 0
				
				# Frame empty
				cv2.putText(frame_empty, "Blinks: {}".format(TOTAL), (30, 60),
							cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
				cv2.putText(frame_empty, "EAR: {:.2f}".format(ear), (30, 90),
							cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
				
				# ********************************
				# convert dlib's rectangle to a OpenCV-style bounding box
				# [i.e., (x, y, w, h)], then draw the face bounding box
				(x, y, w, h) = face_utils.rect_to_bb(rect)
				self.bounding_box = (x, y, w, h)
				# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				
				# Frame empty
				cv2.rectangle(frame_empty, (x, y), (x + w, y + h), (0, 255, 0), 2)
				
				# show the face number
				cv2.putText(frame_empty, "Face #{}".format(i + 1), (30, 120),
							cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
				
				# loop over the (x, y)-coordinates for the facial landmarks
				# and draw them on the image
				for (x, y) in self.shape:
					# cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
					cv2.circle(frame_empty, (x, y), 1, (0, 255, 255), -1)
			
			# **********************************************************
			if frame_got is False:
				break
			
			# If frame comes from webcam, flip it so it looks like a mirror.
			if self.conf.vedio_path == 0:
				frame = cv2.flip(frame, 2)
			
			# Pose estimation by 3 steps:
			# 1. detect face;
			# 2. detect landmarks;
			# 3. estimate pose
			
			# Feed frame to image queue.
			self.img_queue.put(frame)
			
			# Get face from box queue.
			self.facebox = self.box_queue.get()
			
			if self.facebox is not None:
				# Detect landmarks from image of 128x128.
				face_img = frame[self.facebox[1]: self.facebox[3],
						   self.facebox[0]: self.facebox[2]]
				face_img = cv2.resize(face_img, (self.conf.CNN_INPUT_SIZE, self.conf.CNN_INPUT_SIZE))
				face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
				
				self.tm.start()
				# marks = self.mark_detector.detect_marks([face_img])
				self.tm.stop()
				
				# Convert the marks locations from local CNN to global image.
				self.shape *= (self.facebox[2] - self.facebox[0])
				self.shape[:, 0] += self.facebox[0]
				self.shape[:, 1] += self.facebox[1]
				
				# Uncomment following line to show raw marks.
				# mark_detector.draw_marks(
				#     frame, marks, color=(0, 255, 0))
				
				# Uncomment following line to show facebox.
				# mark_detector.draw_box(frame, [facebox])
				
				# Try pose estimation with 68 points.
				self.pose = self.pose_estimator.solve_pose_by_68_points(self.shape)
				
				# Stabilize the pose.
				self.steady_pose = []
				pose_np = np.array(self.pose).flatten()
				for value, ps_stb in zip(pose_np, self.pose_stabilizers):
					ps_stb.update([value])
					self.steady_pose.append(ps_stb.state[0])
				self.steady_pose = np.reshape(self.steady_pose, (-1, 3))
				
				# Uncomment following line to draw pose annotation on frame.
				# pose_estimator.draw_annotation_box(
				# 	frame, pose[0], pose[1], color=(255, 128, 128))
				
				# Uncomment following line to draw stabile pose annotation on frame.
				# pose_estimator.draw_annotation_box(frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))
				
				# Uncomment following line to draw head axes on frame.
				# pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])
				self.pose_estimator.draw_axes(frame_empty, self.steady_pose[0], self.steady_pose[1])
				print('steady pose vector: {}'.format(self.steady_pose[0], self.steady_pose[1]))
			else:
				# cv2.putText(frame, "Signal loss", (200, 200),
				# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.putText(frame_empty, "Signal loss", (200, 200),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# ******************************************************************
			# We send this frame to GazeTracking to analyze it
			self.gaze.refresh(frame)
			
			frame = self.gaze.annotated_frame()
			text = ""
			
			if self.gaze.is_blinking():
				text = "Blinking"
			elif self.gaze.is_right():
				text = "Looking right"
			elif self.gaze.is_left():
				text = "Looking left"
			elif self.gaze.is_center():
				text = "Looking center"
			
			cv2.putText(frame_empty, text, (250, 250), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
			
			left_pupil = self.gaze.pupil_left_coords()
			right_pupil = self.gaze.pupil_right_coords()
			cv2.putText(frame_empty, "Left pupil:  " + str(left_pupil), (250, 280), cv2.FONT_HERSHEY_DUPLEX, 0.5,
						(147, 58, 31), 1)
			cv2.putText(frame_empty, "Right pupil: " + str(right_pupil), (250, 310), cv2.FONT_HERSHEY_DUPLEX, 0.5,
						(147, 58, 31), 1)
			
			# ********************************************************************
			# show the frame
			# cv2.imshow("Frame", frame)
			cv2.imshow("Frame", frame_empty)
			key = cv2.waitKey(1) & 0xFF
			
			self.pass_variable = np.array(1)
			
			try:
				self._listener(self.pass_variable)
			except:
				pass
			
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
		# do a bit of cleanup
		cv2.destroyAllWindows()
		# self.cap.stop()
	
	def set_listener(self, listener):
		self._listener = listener