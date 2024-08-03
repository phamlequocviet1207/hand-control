import cv2
import mediapipe as mp

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

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

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()