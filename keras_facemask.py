"""
Author : Gülşah Büyük
Date : 20.03.2021
"""
import os
import random
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dropout,Dense
import cv2
import numpy as np
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split

data=[]
categories = ["with_mask","without_mask"]
for category in categories:
    path =os.path.join('train',category)
    label=categories.index(category)

    for file in os.listdir(path):
        img_path =os.path.join(path,file)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        data.append([img,label])

# print(len(data))
random.shuffle(data)
X=[]
y=[]
for features,label in data:
    X.append(features)
    y.append(label)
# print(len(X),len(y))

X=np.array(X)
y=np.array(y)
# print(X.shape, y.shape)
# print(y)  only 1,0 as labels are expected
X= X/255 #width and height are 255 for pictures
# print(X[0])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
# print(X_train.shape,X_test.shape)

# model = Sequential([
#     Conv2D(100, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D(2, 2),
#
#     Conv2D(100, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#
#     Flatten(),
#     Dropout(0.5),
#     Dense(50, activation='relu'),
#     Dense(2, activation='sigmoid')
# ])

model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
y_train= y_train.reshape(-1,1)
model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

def detect_face_mask(img):
    y_pred =(model.predict(img.reshape(1,224,224,3)) < 0.5).astype("int32")
    # y_pred =model.predict_classes(img.reshape(1,224,224,3))
    return y_pred[0][0]
    # return y_pred

def draw_label(img,text,pos,bg_color):
    # text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_PLAIN,1,cv2.FILLED)
    # end_x = pos[0] + text_size[0][0] + 2
    # end_y = pos[1] + text_size[0][0] - 2
    # cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,2,cv2.FONT_HERSHEY_PLAIN,bg_color,2,cv2.LINE_AA)

cap=cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
while True:
    ret,frame=cap.read()
    img=cv2.resize(frame,(224,224))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5,minSize=(60,60))
    y_pred = detect_face_mask(img)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    if y_pred == 0:
        draw_label(frame, "Mask", (5,20),(0, 255, 0))
    else:
        draw_label(frame, "No Mask",(5,20),(0, 0, 255))
        # cv2.imshow('Video', frame)
        #
        # if cv2.waitKey(1) & 0xFF ==ord('q'):
        #     break


    # if y_pred == 0:
    #     draw_label(frame, "Mask", (25, 25), (0, 255, 0))
    # else:
    #     draw_label(frame, "No Mask", (25, 25), (0, 0, 255))

    cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    # cv2.imshow('Video', frame)
    # k = cv2.waitKey(30) & 0xFF == ord('q')
    # if k == 27:
    #     break
cap.release()
cv2.destroyAllWindows()








