import cv2

class FaceAndEyeDetection:
    def __init__(self, face_cascade_path, eye_cascade_path):

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        

        if self.face_cascade.empty():
            print("Error: Could not load face cascade.")
            exit()
        if self.eye_cascade.empty():
            print("Error: Could not load eye cascade.")
            exit()

    def detect_faces(self, gray, frame):
        """Detect faces in the given frame."""
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            self.detect_eyes(roi_gray, roi_color)
        return frame

    def detect_eyes(self, roi_gray, roi_color):
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    def start_detection(self):
        video_capture = cv2.VideoCapture(0)


        if not video_capture.isOpened():
            print("Error: Could not access the webcam.")
            exit()

        while True:
            ret, frame = video_capture.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canvas = self.detect_faces(gray, frame)
            cv2.imshow('Face and Eye Detection', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break


        video_capture.release()
        cv2.destroyAllWindows()

face_cascade_path = r'D:\data science\AI\Gen AI\class\10th- Haar cascade classifier\Haarcascades\haarcascade_frontalface_default.xml'
eye_cascade_path = r'D:\data science\AI\Gen AI\class\10th- Haar cascade classifier\Haarcascades\haarcascade_eye.xml'

detection_system = FaceAndEyeDetection(face_cascade_path, eye_cascade_path)

detection_system.start_detection()
