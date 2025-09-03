# face_detector.py
import cv2
import face_recognition
import face_recognition_models
print(face_recognition_models)
print("Starting face detection script...")

# Get a reference to the webcam (0 is usually the built-in webcam)
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")

print("Webcam successfully opened.")
print("Press 'q' to quit.")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # The 'ret' variable is a boolean (True/False) that tells us if the frame was read correctly.
    # The 'frame' variable is the actual image data.
    if not ret:
        break

    # We need to convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Loop through each face found in the frame
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        # We use the original BGR frame because OpenCV's drawing functions expect it.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # The parameters for cv2.rectangle() are:
        # 1. The image to draw on (frame)
        # 2. The top-left corner of the rectangle (left, top)
        # 3. The bottom-right corner of the rectangle (right, bottom)
        # 4. The color of the rectangle (0, 255, 0) which is green in BGR
        # 5. The thickness of the rectangle's border (2 pixels)

    # Display the resulting image with the drawn rectangles
    cv2.imshow('Face Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

print("\nScript finished. Goodbye!")
