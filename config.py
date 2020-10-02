
class config():
	shape_predictor_path = '/home/yizi/Documents/face-api/blink-detection/blink-detection/shape_predictor_68_face_landmarks.dat'
	vedio_path = '/home/yizi/Documents/face-api/faster_facial_landmarks/test_sun.mp4'
	
	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 3
	CNN_INPUT_SIZE = 400
	