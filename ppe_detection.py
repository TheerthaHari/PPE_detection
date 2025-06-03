from ultralytics import YOLO
import cv2

model = YOLO("/home/theertha/Desktop/Yolo/best(1).pt")
cam = cv2.VideoCapture('/dev/video0')

if not cam.isOpened():
    print("Not able to access camera")
    exit()

while True:
    ret, frame=cam.read()
    if not ret:
        print("NO frame captured")
        break

    results = model(frame)
    boxes = results[0].boxes

    detected_id = boxes.cls
    id = detected_id.unique()

    for x in id:
        CLASS_NAME = results[0].names[int(x)]
        print(f"Detected class: {CLASS_NAME} (ID: {int(x)})")

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        color = (0, 255, 0)  

        cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)

        label = f"{results[0].names[cls]}: {conf:.2f}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        y_label = y1 - 20 if y1 - 20 > 0 else y1 + 10

        cv2.rectangle(frame, (x1, y_label), (x1 + w, y_label + h), color, -1)

        cv2.putText(frame, label, (x1, y_label + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("Manual PPE Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
