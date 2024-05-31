import cv2
import face_recognition
import numpy as np

# Load the images of the known people
person_image_encodings = []
person_names = ["Jeeves"]
person_images = ["jee.jpeg"]

for person_image in person_images:
    image = face_recognition.load_image_file(person_image)
    image_encoding = face_recognition.face_encodings(image)[0]
    person_image_encodings.append(image_encoding)
    person_names.append(person_image.split(".")[0])

# Initialize the webcam
video_capture = cv2.VideoCapture(2)

# Get the width and height of the webcam feed
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through the faces found
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the face encoding to the known faces
        results = face_recognition.compare_faces(person_image_encodings, face_encoding)

        # If the face is a match, draw a box around it and display the name
        if True in results:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            name = "Jeeves"
            for i, result in enumerate(results):
                if result:
                    name = person_names[i]
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
video_capture.release()
cv2.destroyAllWindows()