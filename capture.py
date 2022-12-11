import cv2
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import webbrowser
import streamlit as st


# StreamLit Initialization
st.header("Your own's Emotion Based Music Recommender")
capture = st.checkbox('Capture My Emotion now!')
FRAME_WINDOW = st.image([])
captureEmotion = st.button('Capture My Emotion')
tracks = st.text_input(
    'Do you want to listen bollywood songs or western songs')


# casc_path for OpenCV
casc_path = 'haarcascade_frontalface_default.xml'

# model definition
model = Sequential([
    Conv2D(128, (5, 5), input_shape=(48, 48, 1),
           activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Conv2D(256, (5, 5), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    # MaxPool2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(6, activation='softmax')
])
model.load_weights('music_recommendation_model.h5')
emotion_dict = {0: "angry", 1: "fear", 2: "happy",
                3: "sadness", 4: "surprise", 5: "neutral"}


# function to capture users's emotion
def capture_face():
    mood = 'Happy'
    faceCascade = cv2.CascadeClassifier(casc_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            # print(prediction)
            # print(cropped_img[0])
            # print(cropped_img[0].shape)
            # print(type(cropped_img))
            maxindex = int(np.argmax(prediction))
            print(maxindex)
            mood = emotion_dict[maxindex]
            cv2.putText(frame, emotion_dict[maxindex], (x+40, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            faceImage = np.squeeze(cropped_img[0], axis=2).astype(np.uint8)
            cv2.imwrite('faces/face.jpg', faceImage)

        # Display the resulting frame
        # faceImage = cv2.imshow('Video', frame)
        FRAME_WINDOW.image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if captureEmotion:
            # print(captureEmotion)
            recommendSongs(mood)
            break
    cap.release()
    cv2.destroyAllWindows()


# function to recommendSongs
def recommendSongs(userMood):
    webbrowser.open(
        f"https://www.youtube.com/results?search_query=top 10+{tracks}+{userMood}+songs")


if capture:
    capture_face()
