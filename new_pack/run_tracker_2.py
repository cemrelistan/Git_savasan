from siamrpn import TrackerSiamRPN
from redis_helper import RedisHelper
import torch
import ast
import time
import cv2
import threading

class Track:
    def __init__(self):
        net_path = '/home/cello/Desktop/savasan/siamarpn_test/models/model.pth'
        
        try:
            self.tracker = TrackerSiamRPN(net_path=net_path)
        except Exception as e:
            print(f"Tracker yüklenirken hata oluştu: {e}")
            exit(1)

        self.rh = RedisHelper()
        self.last_b_box = None  # Son kullanılan b_box'ı saklamak için
        self.last_update_time = time.time()  # Son b_box kontrol zamanı

    def check_box(self, w, h):
        if w == 48 & h == 27:
            return True
        else:
            return False

    def track(self):
        frame = self.rh.from_redis('frame')
        if frame is None:
            print("Frame Redis'ten alınamadı, çıkılıyor.")
            return

        # İlk başta tespit edilen kutu ile tracker başlat
        b_box = self.rh.r.get("b_box")
        if b_box is None:
            print("Başlangıçta b_box bulunamadı, takip başlatılamıyor.")
            return
        try:
            b_box = ast.literal_eval(b_box.decode('utf-8'))
            x1, y1, x2, y2 = map(int, b_box)
            self.last_b_box = (x1, y1, x2 - x1, y2 - y1)
            print(f"İlk BBox: {self.last_b_box}")
            self.tracker.init(frame, self.last_b_box)
        except ValueError:
            print("Başlangıç b_box formatı hatalı, çıkılıyor.")
            return

        prev_time = time.time()

        while True:
            frame = self.rh.from_redis('frame')
            if frame is None:
                print("Yeni frame alınamadı, bekleniyor...")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
                        
            # **Her 2 saniyede bir yeni `b_box` kontrol et**
            if time.time() - self.last_update_time >= 10:
                new_b_box = self.rh.r.get("b_box")
                if new_b_box is not None:
                    try:
                        new_b_box = ast.literal_eval(new_b_box.decode('utf-8'))
                        x1, y1, x2, y2 = map(int, new_b_box)
                        new_b_box_tuple = (x1, y1, x2 - x1, y2 - y1)

                        # **Eğer yeni b_box geldiyse, tracker'ı güncelle**
                        if new_b_box_tuple != self.last_b_box:
                            print(f"Yeni BBox bulundu, tracker güncelleniyor: {new_b_box_tuple}")
                            self.tracker.init(frame, new_b_box_tuple)
                            self.last_b_box = new_b_box_tuple
                    except ValueError:
                        print("Yeni b_box formatı hatalı, eski kutu ile devam ediliyor.")

                self.last_update_time = time.time()  # Zaman damgasını güncelle

            # **Takip işlemi sürekli devam etsin**
            tracker_box = self.tracker.update(frame)
            if tracker_box is None:
                print("Tracker güncellenemedi, eski kutu ile devam ediliyor.")
                time.sleep(0.1)
                continue

            t_box = [int(v) for v in tracker_box]
            print(f"Yeni takip koordinatları: {t_box}")
            self.rh.r.set("t_box", str(t_box).encode('utf-8'))

            # **KUTUYU ÇİZ VE GÖSTER**
            x, y, w, h = t_box
            color = (0, 255, 0)  # Takip kutusu rengi (yeşil)
            
            if self.last_b_box == (x1, y1, x2 - x1, y2 - y1):
                color = (0, 0, 255)  # Yeni b_box geldiğinde kutu kırmızı olacak
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            if self.check_box(w,h) is True:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Kutuyu çiz
                cv2.putText(frame, "Tracking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Kutuyu çiz
            cv2.putText(frame, "Tracking", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Tracking", frame)  # Görüntüyü ekranda göster
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)  

        cv2.destroyAllWindows()






def run_tracking():
    det = Track()
    det.track()

if __name__ == '__main__':
    detection_thread = threading.Thread(target=run_tracking)
    detection_thread.start()
    detection_thread.join()

