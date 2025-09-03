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
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare the current face with all known faces
            matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.2)
            name = "Unknown" # Default name if no match is found

            # --- Find the best match ---
            if True in matches:
                # Find the indexes of all matched faces
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                
                # Use a dictionary to count votes for each name
                counts = {}
                for i in matched_idxs:
                    name = known_names[i]
                    counts[name] = counts.get(name, 0) + 1
                
                # Determine the name with the most votes
                name = max(counts, key=counts.get)
                
                # Print authentication status to the terminal
                print(f"Authenticate: {name}")

            else:
                 print("Not a valid user")
            
            # --- Draw box and name on the frame ---
            top, right, bottom, left = face_location
            
            # Set color based on authentication status
            if name != "Unknown":
                box_color = (0, 255, 0) # Green for authenticated
            else:
                box_color = (0, 0, 255) # Red for unknown

            # Draw the rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            
            # Draw the name label below the face
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