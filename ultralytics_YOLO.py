import cv2
from ultralytics import YOLO



# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # # Process the frame using the YOLO model
    results = model.track(frame,show=True)

    # # Visualize the results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLO Object Detection', frame)
    cv2.imshow("cam", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam capture and close the window
cap.release()
cv2.destroyAllWindows()