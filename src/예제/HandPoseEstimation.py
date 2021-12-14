import mediapipe as mp
import imutils
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_list = []
cap = cv2.VideoCapture(0)
cnt = 0

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(10) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()