import cv2
import os

video_path = r"C:\Users\ESHA\Documents\vehicle and pedestrian detection Using OpenCV\Dataset (Sample Videos)\my_video.mp4"
car_path = r'C:\Users\ESHA\Documents\vehicle and pedestrian detection Using OpenCV\haarcascades\haarcascade_car.xml'
pedestrian_path = r'C:\Users\ESHA\Documents\vehicle and pedestrian detection Using OpenCV\haarcascades\haarcascade_fullbody.xml'

# Debug prints to make sure paths are correct
print("Video path exists:", os.path.exists(video_path))
print("Car path exists:", os.path.exists(car_path))
print("Pedestrian path exists:", os.path.exists(pedestrian_path))

# Load classifiers
car_tracker = cv2.CascadeClassifier(car_path)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_path)

video = cv2.VideoCapture(video_path)

while True:
    (read_successful, frame) = video.read()

    if not read_successful:
        break

    greyscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_tracker.detectMultiScale(greyscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(greyscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'VEHICLE', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 70), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, 'HUMAN', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Vehicle and Pedestrian Detector', frame)

    if cv2.waitKey(1) in [81, 113]:  # q or Q
        break

video.release()
cv2.destroyAllWindows()