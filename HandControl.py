import cv2
import mediapipe as mp
import math

import numpy as np

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]

vol = 0
volBar = 400
volPer = 0

cap = cv2.VideoCapture(0)

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                       max_num_hands=2) as hands:

# with handsModule.Hands(static_image_mode=True) as hands:
    while True:
        ret, frame = cap.read()

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        imageHeight, imageWidth, _ = frame.shape

        if results.multi_hand_landmarks != None:
            # print(results.multi_hand_landmarks[4])

            for handLandmarks in results.multi_hand_landmarks:
                # for point in handsModule.HandLandmark:
                #     normalizedLandmark = handLandmarks.landmark[point]
                #     pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x,
                #                                                                               normalizedLandmark.y,
                #                                                                               imageWidth, imageHeight)
                #
                #     print(point)
                #     print(pixelCoordinatesLandmark)
                #     print(normalizedLandmark)
                # # print(handLandmarks)

                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)

                if handsModule.HandLandmark(4) and handsModule.HandLandmark(8):
                    point4 = handsModule.HandLandmark(4)
                    point8 = handsModule.HandLandmark(8)

                    normalizedLandmark4 = handLandmarks.landmark[point4]
                    x4,y4 = pixelCoordinatesLandmark4 = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark4.x,
                                                                                              normalizedLandmark4.y,
                                                                                              imageWidth,
                                                                                              imageHeight)

                    # print(x4,y4)

                    normalizedLandmark8 = handLandmarks.landmark[point8]
                    x8,y8 = pixelCoordinatesLandmark8 = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark8.x,
                                                                                              normalizedLandmark8.y,
                                                                                              imageWidth,
                                                                                              imageHeight)

                    cv2.circle(frame, pixelCoordinatesLandmark4, 10, (255,0,255), cv2.FILLED)
                    cv2.circle(frame, pixelCoordinatesLandmark8, 10, (255, 0, 255), cv2.FILLED)


                    # finding the length between 2 points
                    length = math.sqrt((x4-x8)**2 + (y4-y8)**2)
                    # print(length)

                    # Hand Range: 50-300
                    # Volumn Range: -65, 0
                    vol = np.interp(length, [50,300], [minVol,maxVol])
                    volBar = np.interp(length, [50,300], [400,150])
                    volPer = np.interp(length, [50, 300], [0, 100])
                    print(int(length), vol)
                    volume.SetMasterVolumeLevel(vol, None)

                    if length > 50:
                        cv2.line(frame, (x4, y4), (x8, y8), (0, 0, 255), 5)
                    else:
                        cv2.line(frame, (x4, y4), (x8, y8), (0, 255, 0), 5)

            cv2.rectangle(frame, (50,150), (85,400), (0,255,0), 3)
            cv2.rectangle(frame, (50, int(volBar)), (85,400), (0,255,0), cv2.FILLED)
            cv2.putText(frame, f'{int(volPer)}%', (40,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()