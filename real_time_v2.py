"""
실시간 키포인트 기반 수어 예측기(v2, 어텐션)

목적 :
- 웹캠 프레임에서 MediaPipe Holistic으로 키포인트를 추출하여 시퀀스를 구성하고,
  GRU(v2) 모델로 실시간 예측 및 조건부 저장을 수행한다.

입력 형식 :
- 웹캠 스트림, 시퀀스 길이 FRAME_TARGET(기본 90), 특징 차원 152(상대 손 좌표/각도/손-얼굴 관계 등).
- 모델/라벨맵은 카테고리별 저장 파일을 사용한다.

출력 :
- 화면 오버레이 텍스트로 예측 결과/상태 표시.
- CONF_THRESHOLD 이상일 때 `collected_data/카테고리/라벨/구간/시각.npy`로 저장.

조작 :
- 's' : 수집 시작, 'q' : 종료.
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from model_v2 import KeypointGRUModelV2
import pickle
from PIL import ImageFont, ImageDraw, Image
import os
import time

# 1) 설정 : 디바이스/폰트/시퀀스 길이/모델/저장/임계치
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT_PATH = "NanumGothic.ttf"
FONT = ImageFont.truetype(FONT_PATH, 24)
FRAME_TARGET = 90
MODEL_DIR = "models_by_category_mixed_v2"  # v2 학습 모델 폴더
SAVE_DIR = "collected_data"  # 저장할 루트 폴더
CONF_THRESHOLD = 90.0        # 90% 이상만 저장

# 2) 카테고리 선택 : 모델/라벨맵 존재 여부 확인
CATEGORY = input("테스트할 카테고리 이름을 입력하세요: ").strip()
label_map_path = os.path.join(MODEL_DIR, f"{CATEGORY}_label_map.pkl")
model_path = os.path.join(MODEL_DIR, f"{CATEGORY}_model.pth")

if not os.path.exists(label_map_path) or not os.path.exists(model_path):
    print(f"모델 또는 라벨맵이 존재하지 않습니다: {CATEGORY}")
    exit()

with open(label_map_path, "rb") as f:
    label_map = pickle.load(f)
idx_to_label = {v: k for k, v in label_map.items()}

# 3) 모델 로드 : 구조와 클래스 수를 라벨맵에 맞춰 초기화
model = KeypointGRUModelV2(input_dim=152, attn_dim=146, num_classes=len(label_map)).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()
print(f"[{CATEGORY}] v2 모델 로드 완료, 클래스 수: {len(label_map)}")

# 4) Mediapipe 설정 : Holistic으로 전신/얼굴/양손 키포인트 추출
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# 5) 특징 추출 함수 : 상대 좌표/손가락 각도/손-얼굴 관계
def calculate_relative_hand_coords(hand_kpts):
    """손목 기준 상대 좌표로 변환(위치 불변성)"""
    if np.all(hand_kpts == 0): return hand_kpts
    wrist = hand_kpts[0]
    return hand_kpts - wrist

def calculate_finger_angles(hand_kpts):
    """한 손의 관절 각도(라디안) 계산 : 엄지~새끼까지 연속 세 점으로 각도 산출"""
    if np.all(hand_kpts == 0): return np.zeros(10)
    angles = []
    finger_joints = {
        'thumb': [1,2,3,4], 'index':[5,6,7,8], 'middle':[9,10,11,12],
        'ring':[13,14,15,16], 'pinky':[17,18,19,20]
    }
    for joints in finger_joints.values():
        for i in range(len(joints)-2):
            p1, p2, p3 = hand_kpts[joints[i]], hand_kpts[joints[i+1]], hand_kpts[joints[i+2]]
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return np.array(angles)

def calculate_hand_face_relation(lh_kpts, rh_kpts, face_kpts):
    """양 손목과 코(대리점) 간 상대 위치를 연결 특징으로 사용"""
    nose = face_kpts[1] if np.any(face_kpts) else np.zeros(3)
    lw = lh_kpts[0] if np.any(lh_kpts) else np.zeros(3)
    rw = rh_kpts[0] if np.any(rh_kpts) else np.zeros(3)
    return np.concatenate([lw - nose, rw - nose])

def extract_feature(landmarks):
    """Holistic 결과에서 (152,) 특징 벡터와 원본 손 좌표 반환"""
    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in landmarks.pose_landmarks.landmark]) if landmarks.pose_landmarks else np.zeros((33, 4))
    face = np.array([[l.x, l.y, l.z] for l in landmarks.face_landmarks.landmark]) if landmarks.face_landmarks else np.zeros((468, 3))
    lh = np.array([[l.x, l.y, l.z] for l in landmarks.left_hand_landmarks.landmark]) if landmarks.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[l.x, l.y, l.z] for l in landmarks.right_hand_landmarks.landmark]) if landmarks.right_hand_landmarks else np.zeros((21, 3))

    relative_lh = calculate_relative_hand_coords(lh).flatten()
    relative_rh = calculate_relative_hand_coords(rh).flatten()
    angles_lh = calculate_finger_angles(lh)
    angles_rh = calculate_finger_angles(rh)
    rel_feat = calculate_hand_face_relation(lh, rh, face)

    return np.concatenate([relative_lh, relative_rh, angles_lh, angles_rh, rel_feat]), lh, rh

def draw_text(img, text, position):
    """한글 렌더링을 위해 PIL로 텍스트 오버레이"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=FONT, fill=(255, 0, 255))
    return np.array(img_pil)

# 6) 저장 함수 : 신뢰도 구간별로 하위 폴더를 분기해 시퀀스 저장
def save_sequence(sequence, label, confidence):
    """confidence 기준으로 폴더 분류 후 npy로 저장"""
    # 구간별 폴더 이름
    if confidence < 95:
        range_folder = "90_95"
    elif confidence < 100:
        range_folder = "95_100"
    else:
        range_folder = "100"

    save_path = os.path.join(SAVE_DIR, CATEGORY, label, range_folder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{int(confidence)}_{timestamp}.npy"
    np.save(os.path.join(save_path, filename), np.array(sequence))
    print(f"저장 완료: {save_path}/{filename}")

# 7) 실시간 예측 루프 : 수집-예측-저장 파이프라인
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=FRAME_TARGET)
collecting = False
hand_detected = False
result_text = "Press 's' to start"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    if collecting:
        # 7-1) 특징 추출 및 시퀀스 누적
        feature, lh, rh = extract_feature(results)
        if np.any(lh) or np.any(rh): hand_detected = True
        sequence.append(feature)

        text = f"{len(sequence)}/{FRAME_TARGET} 수집 중..."
        frame = draw_text(frame, text, (30, 30))

        if len(sequence) == FRAME_TARGET:
            # 7-2) 길이 도달 시 예측 또는 실패 메시지
            if not hand_detected:
                result_text = "검출된 손 없음"
            else:
                input_tensor = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = model(input_tensor)
                    prob = torch.softmax(logits, dim=-1)
                    pred = torch.argmax(prob, dim=-1).item()
                    confidence = prob[0, pred].item() * 100
                    label = idx_to_label[pred]
                    result_text = f"{label} ({confidence:.1f}%)"

                    # 7-3) 90% 이상이면 nt  저장
                    if confidence >= CONF_THRESHOLD:
                        save_sequence(sequence, label, confidence)

            collecting = False
            sequence.clear()
            hand_detected = False

    # 7-4) 상태 텍스트 오버레이 및 키 이벤트 처리
    frame = draw_text(frame, result_text, (30, 30))
    cv2.imshow("Real-time Sign Recognition (v2)", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        collecting = True
        sequence.clear()
        hand_detected = False
        result_text = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
