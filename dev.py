import cv2
import os
import numpy as np
import time

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image

def register_students(output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the camera
    camera = cv2.VideoCapture(0)  # Change the parameter if you have multiple cameras

    # Prompt user to enter student details
    num_students = int(input("Enter the number of students to register: "))

    for i in range(num_students):
        student_name = input(f"Enter the name of student {i + 1}: ")

        # Create a subdirectory for the student's images
        student_dir = os.path.join(output_dir, student_name)
        os.makedirs(student_dir, exist_ok=True)

        # Capture and preprocess the student's face images
        count = 0
        while count < num_faces_to_capture:
            # Capture frame-by-frame
            ret, frame = camera.read()

            # Detect faces in the frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            # Iterate over the detected faces and capture them
            for j in range(detections.shape[2]):
                confidence = detections[0, 0, j, 2]
                if confidence > confidence_threshold:
                    # Get the bounding box coordinates of the face
                    box = detections[0, 0, j, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (start_x, start_y, end_x, end_y) = box.astype("int")

                    # Crop the face from the frame
                    face = frame[start_y:end_y, start_x:end_x]

                    # Preprocess and save the captured face
                    processed_face = preprocess_image(face)
                    output_path = os.path.join(student_dir, f"face_{count}.jpg")
                    cv2.imwrite(output_path, processed_face * 255.0)  # Convert back to 0-255 range for saving as image

                    count += 1

                    # Draw the bounding box around the face on the frame (optional)
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Register Student', frame)

            # Delay between frames (2 seconds)
            time.sleep(0.5)

            # Check if 'q' key is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera
    camera.release()
    cv2.destroyAllWindows()

# Set the output directory
output_directory = "registered_students"

# Set the confidence threshold for face detection
confidence_threshold = 0.5

# Load the face detection model
model_dir = "model_dir"
caffe_prototxt_path = os.path.join(model_dir, "deploy.prototxt")
caffe_model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Check if the model files exist
if not os.path.exists(caffe_prototxt_path) or not os.path.exists(caffe_model_path):
    print("Error: Could not find the face detection model files.")
    exit()

net = cv2.dnn.readNetFromCaffe(caffe_prototxt_path, caffe_model_path)

# Set the number of faces to capture per student
num_faces_to_capture = 50

# Register students and save their face images
register_students(output_directory)
