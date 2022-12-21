import time

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def getFrames(wnum, seq): # 차례대로 단어 번호, 그 안의 영상 번호
    cap = cv2.VideoCapture(f"../dataset/{str(wnum).zfill(2)}/{str(seq).zfill(2)}_{str(wnum).zfill(2)}.MP4")
    max_len = 150

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []

    ftime = 0
    fgap = 1
    if int(fps) > 15:
        fgap = 2

    while(cap.isOpened()):
        ret, image = cap.read()
        if not ret:
            break

        if ftime % fgap == 0:
            image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frames.append(image)
        ftime += 1

    while(len(frames) < max_len):
        frames.append(0)

    return frames


def facemesh_video(wnum, seq):
    facemesh_payload = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        print(getFrames(wnum, seq)[38])

        for idx, frame in enumerate(getFrames(wnum, seq)):
            # Convert the BGR image to RGB before processing.

            if type(frame) == int:  # 프레임이 존재하지 않으면
                if idx == 0:  # 첫 프레임부터 안 보이면 0 넣는다.
                    facemesh_payload.append(0)
                    continue

                print(len(facemesh_payload))
                print(f'{idx}번째 프레임에는 프레임이 감지되지 않음')
                facemesh_payload.append(facemesh_payload[idx - 1])
                continue

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:  # 혹은 프레임에서 얼굴 감지가 안 되면 (영상이 짧음)
                if idx == 0:  # 첫 프레임부터 안 보이면 0 넣는다.
                    facemesh_payload.append(0)
                    continue

                print(f'{idx}번째 프레임에는 얼굴이 감지되지 않음')
                facemesh_payload.append(facemesh_payload[idx - 1])
                continue

            frame_landmarks = []

            for lm in results.multi_face_landmarks[0].landmark:
                x = lm.x
                y = lm.y
                z = lm.z

                frame_landmarks.append([x, y, z])

            # print(f'{idx}번째 프레임 :', frame)
            # print(f'{idx}번째 프레임, 이 프레임에서 잡히는 랜드마크 갯수 :',len(a[0].landmark))

            #             print(frame_landmarks)
            facemesh_payload.append(frame_landmarks)

        return facemesh_payload


def points_to_displacement(points):
    displacement_payload = []

    # keypoints 개수
    mx = 0
    for frame in points:
        mx = max(mx, len(frame))

    for idx, frame in enumerate(points):
        if type(frame) == int:
            displacement_payload.append([[0, 0, 0] for _ in range(mx)])
            continue

        displacements = []
        for i in range(mx):
            # print(f"{frame[i][0]}-{points[idx][i][0]}")
            displacements.append([frame[i][0] - points[idx - 1][i][0], frame[i][1] - points[idx - 1][i][1],
                                  frame[i][2] - points[idx - 1][i][2]])
        displacement_payload.append(displacements)
    return displacement_payload


def handpose_video(wnum, seq):
    handpose_payload = []  # 0번 : 왼쪽 손, 1번: 오른쪽 손

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:

        for idx, frame in enumerate(getFrames(wnum, seq)):

            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            if type(frame) == int:  # case 1: 프레임이 존재하지 않으면
                if idx == 0:  # 첫 프레임부터 안 보이면 0 넣는다.
                    handpose_payload.append([0, 0])
                    continue
                print(len(handpose_payload))
                print(f'{idx}번째 프레임에는 프레임이 감지되지 않음')
                handpose_payload.append(handpose_payload[idx - 1])
                continue

            results = hands.process(frame)
            print(frame)
            time.sleep(4300)
            print(frame.shape)
            print(results, results.multi_hand_landmarks)
            if not results.multi_hand_landmarks:  # case 2: 프레임은 있는데 손 인식이 아예 안됨
                if idx == 0:
                    handpose_payload.append([0, 0])
                    continue
                print(f'{idx}번째 프레임에는 손이 둘 다 감지되지 않음')
                handpose_payload.append(handpose_payload[idx - 1])
                continue

            handpose_payload.append(handpose_payload[idx - 1])

            for hand in results.multi_handedness:  # case 3: 감지된 손 다 돌면서 처리
                if len(results.multi_handedness) == 1:  # 감지된 손이 1개
                    if hand.classification[0].label == "Left":  # 감지된 한 손이 왼손
                        frame_landmarks_left = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_left.append([x, y, z])
                        handpose_payload[idx][0] = frame_landmarks_left

                    else:  # 감지된 한 손이 오른손
                        frame_landmarks_right = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_right.append([x, y, z])
                        handpose_payload[idx][1] = frame_landmarks_right


                else:  # 두 손 모두 감지
                    if hand.classification[0].label == "Left":  # 감지된 한 손이 왼손
                        print(f'{idx}번째 프레임에서 왼손 감지')
                        frame_landmarks_left = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_left.append([x, y, z])
                        print(f'{len(frame_landmarks_left)}개의 특징점이 왼손에서 감지')
                        handpose_payload[idx][0] = frame_landmarks_left

                    else:  # 감지된 한 손이 오른손
                        print(f'{idx}번째 프레임에서 오른손 감지')
                        frame_landmarks_right = []
                        for lm in results.multi_hand_landmarks[0].landmark:
                            x = lm.x
                            y = lm.y
                            z = lm.z
                            frame_landmarks_right.append([x, y, z])
                        print(f'{len(frame_landmarks_left)}개의 특징점이 오른손에서 감지')
                        handpose_payload[idx][1] = frame_landmarks_right

    return handpose_payload



handpose_video(1, 2)