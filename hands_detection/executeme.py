import cv2
import mediapipe as mp
import time


DRAW_POINTS = "points"
DRAW_LINES = "lines"

mode = DRAW_LINES

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    #None = No hand has been detected
    #Landmark = {
    #   x: float
    #   y: float
    #   z: float
    # }
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            #Show the points
            if mode == DRAW_POINTS:
                mpDraw.draw_landmarks(img,hand)
            elif mode == DRAW_LINES:
                mpDraw.draw_landmarks(img,hand,mpHands.HAND_CONNECTIONS)


    cv2.imshow("Image", img)
    cv2.waitKey(1)