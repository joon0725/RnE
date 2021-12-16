import mediapipe as mp
import imutils
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def getkey(vid=0):
    face_landmarks = []
    left_landmarks = []
    right_landmarks = []
    cap = cv2.VideoCapture(vid)
    cnt = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() or vid != 0:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=1024)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if results is not None:
                if cnt % 3 == 0:
                    if results.face_landmarks is not None and results.right_hand_landmarks is not None and results.left_hand_landmarks is not None:
                        face_landmarks.append(results.face_landmarks.landmark)
                        right_landmarks.append(results.right_hand_landmarks.landmark)
                        left_landmarks.append(results.left_hand_landmarks.landmark)
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
    face_landmarks = [*map(lambda i: [*map(lambda a: {"x": a.x, "y": a.y, "z": a.z}, i)], face_landmarks)]
    left_landmarks = [*map(lambda i: [*map(lambda a: {'x': a.x, 'y': a.y, 'z': a.z}, i)], left_landmarks)]
    right_landmarks = [*map(lambda i: [*map(lambda a: {'x': a.x, 'y': a.y, 'z': a.z}, i)], right_landmarks)]
    a = []
    for n, i in enumerate(face_landmarks):
        a.append([])
        for j in [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80, 81, 82,
                  84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
                  154, 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234, 246, 249, 251, 263,
                  267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317,
                  318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
                  384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466]:
            a[n].append(i[j])

    delta_face_landmarks = [[((a[n][j]['x'] - a[n - 1][j]['x']) ** 2 + (a[n][j]['y'] - a[n - 1][j]['y']) ** 2 +
                              (a[n][j]['z'] - a[n - 1][j]['z']) ** 2) ** (1 / 2) for j in range(1, len(a[n]))] for n in
                            range(1, len(a))]
    print(len(delta_face_landmarks[0]))
    delta_left_landmarks = [[((left_landmarks[n][j]['x'] - left_landmarks[n - 1][j]['x']) ** 2 + (
            left_landmarks[n][j]['y'] - left_landmarks[n - 1][j]['y']) ** 2 + (
                                      left_landmarks[n][j]['z'] - left_landmarks[n - 1][j]['z']) ** 2) ** (1 / 2) for j
                             in range(1, len(left_landmarks[n]))] for n in range(1, len(left_landmarks))]
    print(len(delta_left_landmarks[0]))
    delta_right_landmarks = [[((right_landmarks[n][j]['x'] - right_landmarks[n - 1][j]['x']) ** 2 + (right_landmarks[n][j]['y'] - right_landmarks[n - 1][j]['y']) ** 2 + (
            right_landmarks[n][j]['z'] - right_landmarks[n - 1][j]['z']) ** 2) ** (1 / 2) for j in range(1, len(right_landmarks[n]))] for n in range(1, len(right_landmarks))]
    print(len(delta_right_landmarks[0]))
    cap.release()
    cv2.destroyAllWindows()
    return delta_face_landmarks, delta_left_landmarks, delta_right_landmarks

if __name__=='__main__':
    getkey()