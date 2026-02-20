import cv2
import time

car_classifier_path = r'D:\data science\AI\Gen AI\class\10th- Haar cascade classifier\Haarcascades\haarcascade_car.xml'
car_classifier = cv2.CascadeClassifier(car_classifier_path)

if car_classifier.empty():
    print(f"Error: Could not load the car classifier at {car_classifier_path}. Make sure the path is correct.")
    exit()

video_path = r"D:\data science\AI\Gen AI\project\OpenCv\car video.mp4"

# Load the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open the video at {video_path}. Make sure the file path is correct.")
    exit()

print("Video opened successfully. Starting car detection...")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame or video has ended.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Cars Detection', frame)

    if cv2.waitKey(1) == 13: 
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()