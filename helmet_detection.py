from ultralytics import YOLO
import cv2
#import serial
import time

model = YOLO("runs/detect/train-3/weights/best.pt")
print("Model Loaded Successfully")


#signal = serial.Serial("COM3", 115200)
#time.sleep(2)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Webcam cannot be opened")
    exit()

print("Webcam Opened Successfully")


CONFIDENCE_THRESHOLD = 0.80
DETECTION_TIME = 2.0  # seconds

helmet_start_time = None
head_start_time = None

#previous_status = ""

colors = {
    "head": (0, 0, 255),
    "helmet": (0, 255, 0),
}


while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to capture frame")
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    helmet_detected = False
    head_detected = False

   
    for box in results[0].boxes:

        confidence = float(box.conf[0])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if class_name == "helmet":

            helmet_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.putText(
                frame,
                f"Helmet {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif class_name == "head":

            head_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.putText(
                frame,
                f"Head {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

   
    current_time = time.time()

    if helmet_detected:

        head_start_time = None

        if helmet_start_time is None:
            helmet_start_time = current_time

        elapsed = current_time - helmet_start_time

        if elapsed >= DETECTION_TIME:
            status = "SAFE"
        else:
            status = "CHECKING..."

    elif head_detected:

        helmet_start_time = None

        if head_start_time is None:
            head_start_time = current_time

        elapsed = current_time - head_start_time

        if elapsed >= DETECTION_TIME:
            status = "NO HELMET"
        else:
            status = "CHECKING..."

    else:

        helmet_start_time = None
        head_start_time = None

        status = "NO PERSON"

    
    if status == "SAFE":
        color = (0, 255, 0)

    elif status == "NO HELMET":
        color = (0, 0, 255)

    else:
        color = (0, 255, 255)

    cv2.putText(
        frame,
        status,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

   
   # if status != previous_status:

       # print(status)

        #if status == "SAFE":
            #signal.write(b"SAFE\n")

        #elif status == "NO HELMET":
            #signal.write(b"NO HELMET\n")

        #previous_status = status

   
    cv2.imshow("AI Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print ("Exiting...")

cap.release()
#signal.close()
cv2.destroyAllWindows()