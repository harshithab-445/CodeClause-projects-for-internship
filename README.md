# CodeClause-projects-for-internship
This include few projects that are required to complete my internship. So I'm sharing it here.
Project name= Build a simple image recognition system using OpenCV.

import cv2
import face_recognition
import numpy as np

# --------------------------------------------------------------------------------------------------
# STEP 1: LOAD AND ENCODE KNOWN FACES
# This is the "training" part of our simple system. We load images of people we want to recognize
# and generate a unique 128-dimensional encoding for each face.
# IMPORTANT: Replace these with paths to your own images.
# --------------------------------------------------------------------------------------------------

print("Loading and encoding known faces...")
known_face_encodings = []
known_face_names = []

# Example: Load and encode a photo of a person named 'Jane'.
# Make sure 'jane_doe.jpg' is in the same directory as this script.
try:
    jane_image = face_recognition.load_image_file("jane_doe.jpg")
    jane_face_encoding = face_recognition.face_encodings(jane_image)[0]
    known_face_encodings.append(jane_face_encoding)
    known_face_names.append("Jane")
except (IOError, IndexError):
    print("Warning: Could not find or encode 'jane_doe.jpg'.")

# Example: Load and encode another person named 'John'.
try:
    john_image = face_recognition.load_image_file("john_doe.jpg")
    john_face_encoding = face_recognition.face_encodings(john_image)[0]
    known_face_encodings.append(john_face_encoding)
    known_face_names.append("John")
except (IOError, IndexError):
    print("Warning: Could not find or encode 'john_doe.jpg'.")

if not known_face_encodings:
    print("Error: No known faces loaded. Please add image files and try again.")
    exit()

# Initialize variables for the video stream
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam started. Press 'q' to exit.")
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # We process every other frame to save processing time
        if process_this_frame:
            # Resize the frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # Compare the current face encoding with the known face encodings
                # 'tolerance' determines how strict the match is (lower = more strict)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                # If a match is found, use the name of the first matching person
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up the face locations since the frame was resized
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face and display the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Show the resulting image
        cv2.imshow('Face Recognition System', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam and close windows
cap.release()
cv2.destroyAllWindows()

