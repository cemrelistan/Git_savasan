from ultralytics import YOLO
from redis_helper import RedisHelper
import json
import cv2
import time
import ast  # Redis'ten gelen string'i listeye çevirmek için

class Detection:
    def __init__(self):
        self.model = YOLO("/home/cello/Desktop/savasan/Yarisma_Son/Models/v10son.pt").to("cuda")
        self.rh = RedisHelper()
        self.r = self.rh.r

    def detect(self):
        while True:
            frame = self.rh.from_redis('frame')
            if frame is None:
                print("Frame Redis'ten alınamadı, tekrar deniyor...")
                time.sleep(0.1)
                continue

            results = self.model(frame)
            detections = results[0].boxes.data.cpu().numpy()

            if len(detections) > 0:
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = detection
                    b_box = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Redis'e kaydet
                    self.r.set("b_box", json.dumps(b_box))
                    print(f"Tespit edildi ve Redis'e gönderildi: {b_box}")

                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Yeşil kutu çiz

            else:
                print("Nesne bulunamadı, aramaya devam ediliyor...")

            # Görüntüyü göster
            # cv2.imshow("YOLO Tespit", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basınca çık
            #     break

            time.sleep(0.1)  # Gereksiz yüklenmeyi önlemek için kısa bir bekleme süresi

        # cv2.destroyAllWindows()

if __name__ == '__main__':
    det = Detection()
    det.detect()


        

 

# video_path = "/home/cello/Desktop/savasan/videolar/İtu_takip.mp4" 
# cap = cv2.VideoCapture(video_path)

# ret, frame = cap.read()

# while cap.isOpened():
#     ret, frame = cap.read()
#     print(5)
#     if not ret:
#         break


#     results = model(frame)
#     detections = results[0].boxes.data.cpu().numpy()  # Tespitleri numpy formatına dönüştür

#     for detection in detections:
#         x1, y1, x2, y2, conf, cls = detection
#         b_box = [int(x1), int(y1), int(x2), int(y2)]

#     if closest_box is not None:
#         x1, y1, x2, y2 = closest_box
#         cv2.rectangle(frame, (x1 -5, y1 -5), (x2 +5, y2 +5), (0, 255, 0), 2)


#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break




