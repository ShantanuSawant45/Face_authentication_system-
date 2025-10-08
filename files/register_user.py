# register_user.py

print("Script started!")
try:
    import cv2
    import pickle
    import os
    import face_recognition
    print("Starting user registration...")
    # --- Configuration ---
    ENCODINGS_FILE = "encodings.pkl"
    NUM_IMAGES = 5  # Number of images to capture per user
    # --- Load or Initialize Encodings ---
    # This part loads existing encodings or creates a new file if one doesn't exist
    if os.path.exists(ENCODINGS_FILE):
        print("Loading existing encodings...")
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        print("No existing encodings found. Creating a new database.")
        data = {"encodings": [], "names": []}

    # --- Get New User's Name ---
    # We ensure the name is a valid directory name (no spaces, etc.)
    user_name = input("Enter the new user's name (no spaces): ").lower()

    # --- Liveness pre-check ---
    # Try importing as package first, fallback if run as script
    # try:
    #     from files.liveness import check_liveness
    # except Exception:
    #     from liveness import check_liveness
    #
    # print("Running liveness check before registration (blink or head-movement)...")
    # live_ok, live_reason = check_liveness(duration=4.0, required_blinks=1)
    # print(f"Liveness result: {live_ok} - {live_reason}")
    # if not live_ok:
    #     print("Liveness check failed. Aborting registration.")
    #     exit()

    # --- Initialize Webcam ---
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")

    print(f"Webcam opened. Please capture {NUM_IMAGES} images for registration.")
    print("Press 'c' to capture, 'q' to quit.")

    captured = 0
    new_encodings = []  # temporarily hold encodings for this registration
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display instructions on the frame
        cv2.putText(frame, f"Image {captured+1}/{NUM_IMAGES}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to Capture", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to Quit", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Registration', frame)

        key = cv2.waitKey(1) & 0xFF

        # --- Capture Logic ---
        if key == ord('c'):
            print(f"Capturing image {captured+1} for {user_name}...")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find faces in the frame
            boxes = face_recognition.face_locations(rgb_frame, model='hog')

            # We need exactly one face to register a user
            if len(boxes) == 1:
                # Generate the encoding for this capture and store it temporarily
                encoding = face_recognition.face_encodings(rgb_frame, boxes)[0]
                new_encodings.append(encoding)
                captured += 1
                print(f"✅ Captured image {captured} for {user_name}.")

                # If we have not yet captured enough images, continue capturing
                if captured < NUM_IMAGES:
                    # show a quick visual confirmation
                    cv2.putText(frame, f"Captured {captured}/{NUM_IMAGES}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Registration', frame)
                    cv2.waitKey(500)
                    continue

                # --- After collecting NUM_IMAGES encodings: check duplicates against existing DB ---
                is_duplicate = False
                if data.get("encodings"):
                    try:
                        # For each new encoding, compare against all saved encodings; if any match is found, mark duplicate
                        for new_enc in new_encodings:
                            matches = face_recognition.compare_faces(data["encodings"], new_enc, tolerance=0.5)
                            if any(matches):
                                is_duplicate = True
                                break
                    except Exception:
                        is_duplicate = False

                if is_duplicate:
                    print("❌ This face is already registered. Registration aborted.")
                    cv2.putText(frame, "Already registered", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Registration', frame)
                    cv2.waitKey(2000)
                    break

                # No duplicates found: persist all new encodings and names
                for enc in new_encodings:
                    data["encodings"].append(enc)
                    data["names"].append(user_name)

                # Save the updated data back to the file
                with open(ENCODINGS_FILE, "wb") as f:
                    f.write(pickle.dumps(data))

                print(f"✅ Success! User '{user_name}' has been registered with {NUM_IMAGES} images.")
                cv2.putText(frame, "Success! Registered.", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow('Registration', frame)
                cv2.waitKey(2000)  # Wait for 2 seconds
                break
            else:
                print("❌ Error: Could not find exactly one face. Please try again.")
                
        # --- Quit Logic ---
        elif key == ord('q'):
            print("Registration cancelled.")
            break

    # --- Cleanup ---
    video_capture.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()