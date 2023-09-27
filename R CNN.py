import cv2

# Load the pre-trained Faster R-CNN model for face detection
net = cv2.dnn.readNet('faster_rcnn_resnet50_coco.pb')

# Load the pre-trained Faster R-CNN model configuration file
config_path = 'faster_rcnn_resnet50_coco.config'
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# Open a video capture object
cap = cv2.VideoCapture('footage/footage5.mp4')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame for processing
    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Pass the blob through the network to perform face detection
    net.setInput(blob)
    detections = net.forward()

    # Loop through the detected faces and draw rectangles
    for i in range(detections.shape[2]):
        class_id = int(detections[0, 0, i, 1])
        if class_id == 1:  # Class ID 1 represents 'person'
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Adjust confidence threshold as needed
                box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                (startX, startY, endX, endY) = box.astype(int)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close any OpenCV windows
cv2.destroyAllWindows()
