# Facial Emotion Based Music Detection Web application

## To run this application, make sure you have installed streamlit on your system.


```console
streamlit run capture.py
```

This will start the streamlit application on the 8051 port.


## Project Description

### Technology Stacks Used

	1. Tensorflow
	2  Streamlit
	3. OpenCV
	4. Python
	5. Matplotlib


### CNN Model is designed with 4 Convolution layers, 3 Maxpool layers and 3 Fully connected layers. This CNN model has given training accuracy of 82% and validation accuracy of around 50%. We added droput to our model to improve accuracy further and achived test accuracy of 50%.


### OpenCV and Streamlit libraries are used to capture facial emotions of user using system's webcam. Once the facial emotions are captured, user can click on "Recommend Me a song" to get the list of recommended Youtube audio/videos of emotion based songs. Users alos have options to provide which type of songs they want to listen, e.g. Bollywood, Western etc.
