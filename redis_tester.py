from redis_helper import RedisHelper
import cv2
import time

video_path = '/home/cello/Desktop/savasan/videolar/İtu_takip.mp4'
cap = cv2.VideoCapture(video_path)
r = RedisHelper()

if not cap.isOpened():
    print("Video dosyası açılamadı! Dosya yolunu kontrol edin.")
    exit(1)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya hata oluştu, çıkılıyor...")
            break
        
        r.toRedis("frame", frame)
        # print("Frame gönderildi!")

        time.sleep(0.02)  # Video akıcılığını sağlamak için küçük bir gecikme

except KeyboardInterrupt:
    print("\nİşlem kullanıcı tarafından durduruldu!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Kaynaklar serbest bırakıldı.")
