"""
통합 파이프라인 스크립트

1) `data/` 폴더를 재귀적으로 탐색하여 모든 영상의 MediaPipe Holistic 키포인트를 프레임 단위로 추출합니다.
   - 결과는 원본 폴더 구조를 보존하여 `holistic_keypoints_data/` 아래에 .npy로 저장합니다.
2) 저장된 키포인트 .npy들을 읽어, 다음 특징을 프레임 단위로 계산하여 `feature_data/`에 저장합니다.
   - 상대 손 좌표(양손, 손목 기준), 손가락 관절 각도(10개×양손), 얼굴(코)→양 손목 상대 벡터(6차원)

사용 예시:
  python process_data.py --data-dir data \
                         --keypoints-dir holistic_keypoints_data \
                         --features-dir feature_data

옵션:
  --skip-extract   1단계(키포인트 추출)를 건너뜀
  --skip-features  2단계(특징 엔지니어링)를 건너뜀
  --overwrite      기존 출력이 있어도 덮어쓰기
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from exc


# 1. 기본 설정 및 상수 
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


def list_files_recursively(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    """루트 폴더를 재귀적으로 탐색하여 특정 확장자에 해당하는 파일 경로 목록을 반환합니다."""
    matched_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(exts):
                matched_paths.append(os.path.join(dirpath, filename))
    matched_paths.sort()
    return matched_paths


def ensure_parent_dir(path: str) -> None:
    """파일 경로의 상위 디렉터리를 생성합니다(없으면 생성)."""
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


def extract_video_keypoints(video_path: str, holistic: "mp.solutions.holistic.Holistic") -> np.ndarray:
    """단일 영상에서 프레임별 Holistic 키포인트를 추출하여 (프레임 수, 차원) 형태의 배열로 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    frames_keypoints: List[np.ndarray] = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # 포즈(33, 4: x, y, z, visibility), 얼굴(468, 3), 왼손(21, 3), 오른손(21, 3)
        pose = (
            np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            if results.pose_landmarks is not None
            else np.zeros((33, 4), dtype=np.float32)
        )
        face = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32)
            if results.face_landmarks is not None
            else np.zeros((468, 3), dtype=np.float32)
        )
        left_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            if results.left_hand_landmarks is not None
            else np.zeros((21, 3), dtype=np.float32)
        )
        right_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            if results.right_hand_landmarks is not None
            else np.zeros((21, 3), dtype=np.float32)
        )

        # 하나의 1D 벡터로 결합하여 프레임 리스트에 추가
        frame_keypoints = np.concatenate([
            pose.reshape(-1),
            face.reshape(-1),
            left_hand.reshape(-1),
            right_hand.reshape(-1),
        ])
        frames_keypoints.append(frame_keypoints)

    cap.release()

    if len(frames_keypoints) == 0:
        return np.zeros((0, 33 * 4 + 468 * 3 + 21 * 3 + 21 * 3), dtype=np.float32)

    return np.stack(frames_keypoints).astype(np.float32)


def extract_holistic_for_all_videos(data_dir: str, keypoints_dir: str, overwrite: bool = False) -> None:
    """1단계: `data_dir`의 모든 영상을 처리하여 Holistic 키포인트 .npy를 `keypoints_dir`에 저장합니다."""
    video_paths = list_files_recursively(data_dir, VIDEO_EXTENSIONS)
    if not video_paths:
        print(f"No videos found in '{data_dir}'. Supported: {VIDEO_EXTENSIONS}")
        return

    print(f"Found {len(video_paths)} videos under '{data_dir}'. Extracting keypoints → '{keypoints_dir}'")

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for video_path in tqdm(video_paths, desc="Extracting Holistic"):
            # 원본 경로 구조를 보존하여 .npy 저장 경로 구성
            rel_path = os.path.relpath(video_path, data_dir)
            save_path = os.path.join(
                keypoints_dir,
                os.path.splitext(rel_path)[0] + ".npy",
            )

            if not overwrite and os.path.exists(save_path):
                continue

            keypoints_seq = extract_video_keypoints(video_path, holistic)
            ensure_parent_dir(save_path)
            np.save(save_path, keypoints_seq)

    print(f"Keypoint extraction complete → '{keypoints_dir}'")


# 2. 특징 엔지니어링(Feature Engineering)

def calculate_relative_hand_coords(hand_kpts: np.ndarray) -> np.ndarray:
    """손목(0번) 기준으로 손 랜드마크의 상대 좌표를 계산합니다."""
    if hand_kpts.size == 0 or np.all(hand_kpts == 0):
        return hand_kpts
    wrist = hand_kpts[0]
    return (hand_kpts - wrist).astype(np.float32)


def calculate_finger_angles(hand_kpts: np.ndarray) -> np.ndarray:
    """손가락 관절 각도 10개(엄지 2, 검지·중지·약지·소지 각 2)를 라디안 값으로 계산합니다."""
    if hand_kpts.size == 0 or np.all(hand_kpts == 0):
        return np.zeros(10, dtype=np.float32)

    angles: List[float] = []
    finger_joints = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    for _, joints in finger_joints.items():
        for i in range(len(joints) - 2):
            p1 = hand_kpts[joints[i]]
            p2 = hand_kpts[joints[i + 1]]
            p3 = hand_kpts[joints[i + 2]]

            v1 = p1 - p2
            v2 = p3 - p2

            denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = float(np.dot(v1, v2) / denom)
            cos_angle = max(min(cos_angle, 1.0), -1.0)
            angle = float(np.arccos(cos_angle))
            angles.append(angle)

    return np.array(angles, dtype=np.float32)


def calculate_hand_face_relation(lh_kpts: np.ndarray, rh_kpts: np.ndarray, face_kpts: np.ndarray) -> np.ndarray:
    """얼굴(코, face index 1)에서 양 손목까지의 상대 위치 벡터(총 6차원)를 계산합니다."""
    nose = face_kpts[1] if face_kpts.size != 0 and np.any(face_kpts) else np.zeros(3, dtype=np.float32)
    left_wrist = lh_kpts[0] if lh_kpts.size != 0 and np.any(lh_kpts) else np.zeros(3, dtype=np.float32)
    right_wrist = rh_kpts[0] if rh_kpts.size != 0 and np.any(rh_kpts) else np.zeros(3, dtype=np.float32)
    vec_face_to_lh = left_wrist - nose
    vec_face_to_rh = right_wrist - nose
    return np.concatenate([vec_face_to_lh, vec_face_to_rh]).astype(np.float32)


def compute_features_from_sequence(holistic_sequence: np.ndarray) -> np.ndarray:
    """하나의 영상 키포인트 시퀀스에서 프레임별 152차원 특징을 계산하여 (프레임 수, 152) 배열로 반환합니다."""
    if holistic_sequence.size == 0:
        return holistic_sequence

    engineered_frames: List[np.ndarray] = []

    for frame_data in holistic_sequence:
        # 33*4 | 468*3 | 21*3 | 21*3 순서로 저장된 1D 벡터를 부위별로 복원
        pose = frame_data[0 : 33 * 4].reshape(33, 4)
        face = frame_data[33 * 4 : 33 * 4 + 468 * 3].reshape(468, 3)
        lh = frame_data[33 * 4 + 468 * 3 : 33 * 4 + 468 * 3 + 21 * 3].reshape(21, 3)
        rh = frame_data[33 * 4 + 468 * 3 + 21 * 3 :].reshape(21, 3)

        relative_lh = calculate_relative_hand_coords(lh)
        relative_rh = calculate_relative_hand_coords(rh)
        angles_lh = calculate_finger_angles(lh)
        angles_rh = calculate_finger_angles(rh)
        relation_face_hands = calculate_hand_face_relation(lh, rh, face)

        frame_features = np.concatenate([
            relative_lh.reshape(-1),  # 63
            relative_rh.reshape(-1),  # 63
            angles_lh,                # 10
            angles_rh,                # 10
            relation_face_hands,      # 6
        ]).astype(np.float32)         # 총 152차원

        engineered_frames.append(frame_features)

    return np.stack(engineered_frames).astype(np.float32)


def compute_features_for_all_npy(keypoints_dir: str, features_dir: str, overwrite: bool = False) -> None:
    """2단계: 키포인트 .npy 전체를 읽어 프레임별 152차 특징을 계산하고 `features_dir`에 저장합니다."""
    npy_paths: List[str] = []
    for dirpath, _, filenames in os.walk(keypoints_dir):
        for filename in filenames:
            if filename.lower().endswith(".npy"):
                npy_paths.append(os.path.join(dirpath, filename))
    npy_paths.sort()

    if not npy_paths:
        print(f"No .npy keypoint files found in '{keypoints_dir}'.")
        return

    print(f"Found {len(npy_paths)} keypoint files. Computing features → '{features_dir}'")

    for npy_path in tqdm(npy_paths, desc="Engineering Features"):
        # 키포인트 파일의 상대 경로를 그대로 사용하여 특징 저장 경로 구성
        rel_path = os.path.relpath(npy_path, keypoints_dir)
        save_path = os.path.join(features_dir, rel_path)

        if not overwrite and os.path.exists(save_path):
            continue

        try:
            holistic_sequence = np.load(npy_path)
        except Exception as exc:
            print(f"Failed to load '{npy_path}': {exc}")
            continue

        features_sequence = compute_features_from_sequence(holistic_sequence)
        ensure_parent_dir(save_path)
        np.save(save_path, features_sequence)

    print(f"Feature engineering complete → '{features_dir}'")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """CLI 인자들을 정의하고 파싱하여 반환합니다."""
    parser = argparse.ArgumentParser(description="Video → Holistic keypoints → Engineered features pipeline")
    parser.add_argument("--data-dir", default="data", help="Root folder containing input videos (scanned recursively)")
    parser.add_argument("--keypoints-dir", default="holistic_keypoints_data", help="Output folder for keypoint .npy files")
    parser.add_argument("--features-dir", default="feature_data", help="Output folder for engineered feature .npy files")
    parser.add_argument("--skip-extract", action="store_true", help="Skip keypoint extraction step")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature engineering step")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """엔드 투 엔드 실행 함수: 1) 키포인트 추출 → 2) 특징 엔지니어링 순으로 실행합니다."""
    args = parse_args(argv)

    # 1. 키포인트 추출 
    if not args.skip_extract:
        extract_holistic_for_all_videos(
            data_dir=args.data_dir,
            keypoints_dir=args.keypoints_dir,
            overwrite=args.overwrite,
        )
    else:
        print("Skipping keypoint extraction step.")

    # 2. 특징 엔지니어링 
    if not args.skip_features:
        compute_features_for_all_npy(
            keypoints_dir=args.keypoints_dir,
            features_dir=args.features_dir,
            overwrite=args.overwrite,
        )
    else:
        print("Skipping feature engineering step.")


if __name__ == "__main__":
    main(sys.argv[1:])


