# encode_faces.py
print("Script started!") 
try:
    import face_recognition
    import pickle
    import cv2
    import os

    print("Starting to encode faces...")

    # Path to the directory of images of known people
    dataset_path = "dataset"

    # Initialize the lists to store encodings and names
    known_encodings = []
    known_names = []

    # Loop over the folders in the dataset
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        # Check if it's a directory
        if not os.path.isdir(person_path):
            continue

        # Loop over the images in the person's folder
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)

            # Load the image and convert it from BGR (OpenCV default) to RGB
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.")
                continue
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect the face
            boxes = face_recognition.face_locations(rgb_image, model='hog')

            # Compute the facial embedding for the face
            # We assume there is only ONE face per image for the dataset
            encodings = face_recognition.face_encodings(rgb_image, boxes)

            # Add each encoding + name to our lists
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_name)
                print(f"Encoded image for {person_name}")

    # Save the facial encodings and names to a file
    print("\nSaving encodings to file...")
    data = {"encodings": known_encodings, "names": known_names}
    with open("encodings.pkl", "wb") as f:
        f.write(pickle.dumps(data))

    print("Encodings saved successfully. You can now run the recognition script.")
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()