# register_user.py - With Anti-Spoofing

print("Script started!")
try:
    import sys
    import os

    # Add the Silent-Face-Anti-Spoofing directory to Python path
    # CHANGE THIS PATH TO YOUR ACTUAL PATH
    ANTI_SPOOF_PATH = r"D:\face_detection\Silent-Face-Anti-Spoofing"
    sys.path.insert(0, ANTI_SPOOF_PATH)

    import cv2
    import pickle
    import face_recognition
    import numpy as np
    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name

    print("Starting user registration with anti-spoofing...")

    # --- Configuration ---
    ENCODINGS_FILE = "encodings.pkl"
    NUM_IMAGES = 5  # Number of images to capture per user

    # Use absolute paths based on Anti-Spoof project location
    BASE_PATH = r"D:\face_detection\Silent-Face-Anti-Spoofing"
    MODEL_DIR = os.path.join(BASE_PATH, "resources", "anti_spoof_models")

    DEVICE_ID = 0  # GPU device ID (0 for first GPU, -1 for CPU)
    SPOOF_THRESHOLD = 0.5  # Threshold for real face classification

    # Change working directory to Anti-Spoof project root
    # This is needed for the face detector to find its config files
    original_dir = os.getcwd()
    os.chdir(BASE_PATH)

    # --- Initialize Anti-Spoofing Models ---
    print("Loading anti-spoofing models...")
    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()

    # --- Load or Initialize Encodings ---
    # Use absolute path for encodings file in the files directory
    encodings_path = os.path.join(original_dir, ENCODINGS_FILE)
    if os.path.exists(encodings_path):
        print("Loading existing encodings...")
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
    else:
        print("No existing encodings found. Creating a new database.")
        data = {"encodings": [], "names": []}

    # --- Get New User's Name ---
    user_name = input("Enter the new user's name (no spaces): ").lower()


    # --- Function to Check if Face is Real ---
    def is_real_face(frame):
        """
        Check if the face in frame is real using anti-spoofing models.
        Returns: (is_real, confidence_score)
        """
        # Get face bounding box
        image_bbox = model_test.get_bbox(frame)

        # Check if face detected
        if image_bbox[0] == 0 and image_bbox[1] == 0 and image_bbox[2] == 0 and image_bbox[3] == 0:
            return False, 0.0

        prediction = np.zeros((1, 3))

        # Run prediction with all models
        for model_name in os.listdir(MODEL_DIR):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": frame,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)

            prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))

        # Get label and confidence
        label = np.argmax(prediction)
        confidence = prediction[0][label] / 2

        # Label 1 = Real Face
        is_real = (label == 1 and confidence >= SPOOF_THRESHOLD)

        return is_real, confidence


    # --- Initialize Webcam ---
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise IOError("Cannot open webcam")

    print(f"Webcam opened. Please capture {NUM_IMAGES} images for registration.")
    print("⚠️  Anti-spoofing check will be performed before registration.")
    print("Press 'c' to capture, 'q' to quit.")

    captured = 0
    new_encodings = []
    spoof_check_passed = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display instructions on the frame
        cv2.putText(frame, f"Image {captured + 1}/{NUM_IMAGES}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to Capture", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to Quit", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show anti-spoofing status
        if captured == 0:
            cv2.putText(frame, "Anti-Spoof: Waiting...", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Registration', frame)

        key = cv2.waitKey(1) & 0xFF

        # --- Capture Logic ---
        if key == ord('c'):
            print(f"Capturing image {captured + 1} for {user_name}...")

            # --- ANTI-SPOOFING CHECK (First capture only) ---
            if captured == 0 and not spoof_check_passed:
                print("🔍 Performing anti-spoofing check...")
                cv2.putText(frame, "Checking for spoofing...", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Registration', frame)
                cv2.waitKey(100)

                is_real, spoof_confidence = is_real_face(frame)

                if not is_real:
                    print(f"❌ SPOOFING DETECTED! Confidence: {spoof_confidence:.2f}")
                    print("Registration aborted. Please use a real face.")
                    cv2.putText(frame, "FAKE FACE DETECTED!", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(frame, "Registration Aborted", (50, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow('Registration', frame)
                    cv2.waitKey(3000)
                    break
                else:
                    print(f"✅ Real face verified! Confidence: {spoof_confidence:.2f}")
                    print("Proceeding with registration...")
                    spoof_check_passed = True
                    cv2.putText(frame, f"REAL FACE: {spoof_confidence:.2f}", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Registration', frame)
                    cv2.waitKey(1000)

            # --- Face Recognition Encoding ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame, model='hog')

            if len(boxes) == 1:
                encoding = face_recognition.face_encodings(rgb_frame, boxes)[0]
                new_encodings.append(encoding)
                captured += 1
                print(f"✅ Captured image {captured} for {user_name}.")

                if captured < NUM_IMAGES:
                    cv2.putText(frame, f"Captured {captured}/{NUM_IMAGES}", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow('Registration', frame)
                    cv2.waitKey(500)
                    continue

                # --- Check for Duplicates ---
                is_duplicate = False
                if data.get("encodings"):
                    try:
                        for new_enc in new_encodings:
                            matches = face_recognition.compare_faces(data["encodings"], new_enc, tolerance=0.5)
                            if any(matches):
                                is_duplicate = True
                                break
                    except Exception:
                        is_duplicate = False

                if is_duplicate:
                    print("❌ This face is already registered. Registration aborted.")
                    cv2.putText(frame, "Already registered", (50, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Registration', frame)
                    cv2.waitKey(2000)
                    break

                # --- Save Encodings ---
                for enc in new_encodings:
                    data["encodings"].append(enc)
                    data["names"].append(user_name)

                with open(encodings_path, "wb") as f:
                    f.write(pickle.dumps(data))

                print(f"✅ Success! User '{user_name}' has been registered with {NUM_IMAGES} images.")
                cv2.putText(frame, "Success! Registered.", (50, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow('Registration', frame)
                cv2.waitKey(2000)
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

    # Restore original directory
    os.chdir(original_dir)

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback

    traceback.print_exc()