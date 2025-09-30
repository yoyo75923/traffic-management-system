import cv2
# (Load the video capture device)
cap = cv2.VideoCapture('traffic_video.mp4') #You can add live capture feed of camera module using Android API:- https://shorturl.at/fMBjd
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment out the vehicles
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the vehicles
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and count the number of vehicles
    vehicle_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            vehicle_count += 1

    print(f'Vehicle count: {vehicle_count}')
