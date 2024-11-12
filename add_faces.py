import cv2
import pickle
import numpy as np
import os

# Ensure the 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Check if the CascadeClassifier loaded correctly
if facedetect.empty():
    print("Error loading cascade classifier. Check the file path.")
    exit()

faces_data = []
i = 0
name = input("Enter Your Name: ")

# Collect face data
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Collect a face every 10 frames
        if len(faces_data) < 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1

        # Display count and rectangle on frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

# Release video and close windows
video.release()
cv2.destroyAllWindows()

# Reshape face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)  # Adjust dynamically

# Save names and faces data to pickle files
names_file = 'data/names.pkl'
faces_file = 'data/faces_data.pkl'

# Save names
if not os.path.exists(names_file):
    names = [name] * len(faces_data)
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * len(faces_data)

with open(names_file, 'wb') as f:
    pickle.dump(names, f)

# Save face data
if not os.path.exists(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)

    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)
