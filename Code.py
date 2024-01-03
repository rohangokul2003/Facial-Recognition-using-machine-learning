#!/usr/bin/env python
# coding: utf-8
import cv2
import face_recognition

# Load the reference image
ref_image = face_recognition.load_image_file("reference.jpg")
ref_encoding = face_recognition.face_encodings(ref_image)[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Find face locations in the current frame
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) > 0:
        # Encode the first detected face in the frame
        frame_encoding = face_recognition.face_encodings(frame, face_locations)[0]

        # Compare the face encoding with the reference encoding
        match = face_recognition.compare_faces([ref_encoding], frame_encoding)

        if match[0]:
            text = "Match Found"
            color = (0, 255, 0)  # Green
        else:
            text = "Match Not Found"
            color = (0, 0, 255)  # Red

        # Draw a rectangle around the detected face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display the result on the frame
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
