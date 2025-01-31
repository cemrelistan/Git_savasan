import cv2
from ultralytics import YOLO


model = YOLO("/home/cello/Desktop/savasan/Yarisma_Son/Models/v10son.pt")

video_path = "/home/cello/Desktop/savasan/videolar/İtu_takip.mp4" 
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    print(5)
    if not ret:
        break


    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()  # Tespitleri numpy formatına dönüştür

    closest_box = None

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        closest_box = [int(x1), int(y1), int(x2), int(y2)]

    if closest_box is not None:
        x1, y1, x2, y2 = closest_box
        cv2.rectangle(frame, (x1 -5, y1 -5), (x2 +5, y2 +5), (0, 255, 0), 2)


    cv2.imshow("YOLO Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()



