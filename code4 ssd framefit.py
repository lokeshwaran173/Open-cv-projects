import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier('capgemini/haarcascade_frontalface_default.xml')

# Load the pre-trained SSD model for face detection
net = cv2.dnn.readNetFromCaffe('capgemini/deploy.prototxt.txt', 'capgemini/res10_300x300_ssd_iter_140000.caffemodel')

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

# Define the desired downscale factor (adjust as needed)
downscale_factor = 0.5  # Reduce frame size by half

# Calculate the new frame dimensions
downscaled_width = int(frame_width * downscale_factor)
downscaled_height = int(frame_height * downscale_factor)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Downscale the frame
    frame = cv2.resize(frame, (downscaled_width, downscaled_height))

    # Convert the frame to grayscale for Haar Cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform Haar Cascade face detection
    faces_haar = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform SSD-based face detection
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104, 117, 123))
    net.setInput(blob)
    detections_ssd = net.forward()

    # Draw rectangles around the faces detected by Haar Cascade
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw rectangles around the faces detected by SSD
    for i in range(detections_ssd.shape[2]):
        confidence = detections_ssd[0, 0, i, 2]
        if confidence > 0.5:  # Adjust this threshold
            box = detections_ssd[0, 0, i, 3:7] * [downscaled_width, downscaled_height, downscaled_width, downscaled_height]
            (startX, startY, endX, endY) = box.astype(int)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # Resize the frame for display
    frame = cv2.resize(frame, (display_width, display_height))

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Reduce the delay to increase the speed
    delay = 1  # Adjust this value for faster playback

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
