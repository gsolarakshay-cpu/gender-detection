import cv2 as cv
from deepface import DeepFace as dpf

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from the webcam
capture = cv.VideoCapture(0)

while True:
    success, img = capture.read()

    if not success:
        break

    # Convert the image to grayscale as the classifier works on grayscale images
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region for DeepFace analysis
        face_img = img[y:y+h, x:x+w]
        
        # Analyze the face using DeepFace
        results = dpf.analyze(face_img, actions=("gender", "age"), enforce_detection=False)

        # Get the results
        Age = results[0]["age"]
        Gender = results[0]['dominant_gender']
        name = f'Gender:{Gender}, Age:{Age}'

        # Draw rectangles around detected faces
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.rectangle(img, (x, y - 35), (x + w, y), (0, 255, 0), cv.FILLED)
        cv.putText(img, name, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image with rectangles
    cv.imshow('webcam', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
capture.release()
cv.destroyAllWindows()
