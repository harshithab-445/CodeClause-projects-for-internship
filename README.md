# CodeClause-projects-for-internship
This include few projects that are required to complete my internship. So I'm sharing it here.
Project name= Build a simple image recognition system using OpenCV.

import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam started. Press 'q' to exit.")
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        # The detectMultiScale function detects objects of different sizes in the input image.
        # It returns a list of rectangles where it believes it found the faces.
        faces = face_cascade.detectMultiScale(gray_frame, 
                                             scaleFactor=1.1,  # How much the image size is reduced at each image scale
                                             minNeighbors=5,   # How many neighbors each candidate rectangle should have
                                             minSize=(30, 30)) # Minimum possible object size

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
