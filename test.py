import cv2
import pyttsx3
import time
from ultralytics import YOLO

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 말 속도
engine.setProperty('voice', 'com.apple.speech.synthesis.voice.yuna')  # Mac에서 한국어 음성

model = YOLO('best.pt')

last_announced = {}
cooldown = 5

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        conf = float(box.conf[0])

        now = time.time()
        if name not in last_announced or now - last_announced[name] > cooldown:
            print(f"감지된 객체: {name}, 신뢰도: {conf:.2f}")
            engine.say(f"{name} 입니다")
            engine.runAndWait()
            last_announced[name] = now

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{name} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()