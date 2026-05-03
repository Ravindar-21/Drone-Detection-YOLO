from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()
    
    # Check if drone is detected (class name = 'drone')
    drone_detected = False
    for cls in results[0].boxes.cls:
        class_name = model.names[int(cls)]
        if class_name.lower() == 'drone':
            drone_detected = True
            break
    
    if drone_detected:
        cv2.putText(annotated_frame, "DRONE DETECTED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("YOLO Webcam Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
