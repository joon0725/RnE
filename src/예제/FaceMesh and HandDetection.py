import mediapipe as mp
import imutils
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
right_hand_list = []
left_hand_list = []
cap = cv2.VideoCapture(0)
cnt = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=1024)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        if cnt % 25 == 0:
            right_hand_list.append(results.right_hand_landmarks)
            left_hand_list.append(results.left_hand_landmarks)
            print(f"{cnt/25} {results.face_landmarks}\n\n")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        cv2.imshow('Holistic Model Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cnt += 1

cap.release()
cv2.destroyAllWindows()
print(right_hand_list[8][0])
print(left_hand_list)