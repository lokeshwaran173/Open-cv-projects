import cv2
import numpy as np

# Load YOLO model for person detection
net = cv2.dnn.readNet('capgemini/yolo/yolov3.weights', 'capgemini/yolo/yolov3.cfg')

# Load COCO labels (contains class names)
with open('capgemini/yolo/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Open a video capture object
cap = cv2.VideoCapture('capgemini/footage/footage5.mp4')

# Check if the video capture object is opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Get the original frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the desired display width (adjust this value)
display_width = 1600  # Adjust to your desired width

# Calculate the corresponding display height to maintain the aspect ratio
display_height = int((display_width / frame_width) * frame_height)

# Reduce the display height to fit the screen better
display_height = 900  # Adjust to your desired height

# Define the desired delay (adjust this value for faster playback)
delay = 1  # 1 millisecond per frame (adjust as needed for your desired speed)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Get the frame's blob for YOLO model
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)

    # Set the blob as the input to the YOLO network
    net.setInput(blob)

    # Get detections from YOLO model
    detections = net.forward()

    for detection in detections:
        for obj in detection:
            class_id = int(obj[1])
            confidence = obj[2]

            if confidence > 0.5 and classes[class_id] == 'person':
                center_x = int(obj[0] * frame_width)
                center_y = int(obj[1] * frame_height)
                width = int(obj[2] * frame_width)
                height = int(obj[3] * frame_height)

                # Calculate the top-left and bottom-right coordinates
                top_left_x = int(center_x - width / 2)
                top_left_y = int(center_y - height / 2)
                bottom_right_x = int(center_x + width / 2)
                bottom_right_y = int(center_y + height / 2)

                # Draw a bounding box around the detected person
                cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    # Resize the frame for display
    frame = cv2.resize(frame, (display_width, display_height))

    # Display the frame with detected persons
    cv2.imshow('Person Detection', frame)

    # Reduce the delay to increase the speed
    delay = 1  # Adjust this value for faster playback

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
