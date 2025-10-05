# authenticate.py

print("Script started!")
try:
    import face_recognition
    import pickle
    import cv2

    print("Starting authentication system...")

    # --- Configuration ---
    ENCODINGS_FILE = "encodings.pkl"
    MODEL = "hog"  # or "cnn" for more accuracy (requires dlib with CUDA)

    # --- Load Known Faces ---
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: The encodings file '{ENCODINGS_FILE}' was not found.")
        print("Please run the registration script first.")
        exit()

    known_encodings = data["encodings"]
    known_names = data["names"]

    # --- Initialize Webcam ---
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")

    print("Webcam opened. System is ready.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame, model=MODEL)
        # Generate encodings for the detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        # Use a stricter tolerance and choose the best match by distance
        TOLERANCE = 0.4  # lower = stricter matching (0.4-0.5 is typical)
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name = "Unknown"

            # If no known encodings, skip matching
            if len(known_encodings) > 0:
                # Compute distances to all known encodings and pick the best
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                # distances is a numpy array; get minimum distance
                min_distance = float(distances.min()) if len(distances) > 0 else None
                if min_distance is not None and min_distance <= TOLERANCE:
                    best_match_index = int(distances.argmin())
                    name = known_names[best_match_index]
                    print(f"Authenticate: {name} (distance={min_distance:.3f})")
                else:
                    # optional: debug print the best distance
                    if min_distance is not None:
                        print(f"No match (best distance={min_distance:.3f})")
                    else:
                        print("No known encodings available to match against")

            # --- Draw box and name on the frame ---
            top, right, bottom, left = face_location
            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the final frame
        cv2.imshow('Authentication', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    video_capture.release()
    cv2.destroyAllWindows()
    print("Authentication system shut down.")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()