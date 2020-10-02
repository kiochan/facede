from face_detection import face_detector

def f(data):
	print(data)

def main():
	face = face_detector()
	face.set_listener(f)
	face.detect()

if __name__ == '__main__':
	main()