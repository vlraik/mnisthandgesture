import numpy as np
import cv2
from collections import deque
from keras.models import load_model
cap = cv2.VideoCapture(0)

pts = deque(maxlen=30)

Lower_green = np.array([110, 50, 50])
Upper_green = np.array([130, 255, 255])

model = load_model('mnist_train.h5')

while True:

    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, Lower_green, Upper_green)
    mask = cv2.erode(mask, kernel, iterations=1)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    cnts, heir = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]


    center = None


    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    blackscreen = np.zeros((480, 640, 1), dtype=np.uint8)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thick = 30
        cv2.line(img, pts[i - 1], pts[i], (0, 0, 225), thick)
        cv2.line(blackscreen,pts[i - 1], pts[i], (255, 255, 225), thick)

    blackscreen = cv2.flip(blackscreen,1)
    img = cv2.flip(img,1)
    mnistsized = cv2.resize(blackscreen,(28,28))

    cv2.imshow("MNIST size", mnistsized)
    mnistsized = mnistsized.reshape(1,28,28,1)
    answer = model.predict(mnistsized)
    cv2.putText(img, "Answer: "+str(np.argmax(answer)),(10, 410),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Blackscreen", blackscreen)
    cv2.imshow("Frame", img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        print("Exiting...")
        break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()