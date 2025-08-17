"""
카테고리별 혼합 데이터로 GRU(v2, 어텐션) 모델을 학습하는 스크립트.

목적 :
- 동일 카테고리의 서로 다른 전처리 버전 데이터셋(예: 기본/균등샘플링)을 합쳐 일반화 성능을 향상.

입력 형식 :
- BASE_DIRS 내 각 루트에 카테고리 폴더, 그 하위에 클래스 폴더, .npy 시퀀스 파일 구조를 가정.
- 각 .npy는 (T, F) (예: (90, 152)) 형태.

출력 파일 :
- 카테고리별로 학습된 가중치(.pth)와 해당 카테고리의 라벨 맵(.pkl).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from NpySequenceDataset import NpySequenceDataset
from model_v2 import KeypointGRUModelV2
import pickle

# 1) 경로 설정 : 학습에 사용할 데이터 루트, 저장 경로, 하이퍼파라미터
BASE_DIRS = [
    'data/feature_data_90fps',        # 초기 정제 방식
    'data/feature_data_90fps_v1'      # 균등샘플링 + 마지막 프레임 복제
]
MODEL_SAVE_DIR = 'models_by_category_mixed_v2'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 2) 카테고리별 혼합 학습 : 동일 카테고리를 서로 다른 전처리 버전으로 결합해 학습
for category in sorted(os.listdir(BASE_DIRS[0])):
    category_paths = [os.path.join(base, category) for base in BASE_DIRS]
    
    # 두 폴더 모두 존재해야 학습 진행
    if not all(os.path.isdir(path) for path in category_paths):
        continue

    print(f"[카테고리: {category}] 혼합 학습(v2 Attention) 시작")

    # 3) 라벨맵 생성 : 첫 번째 데이터셋의 클래스 폴더 기준으로 어휘 사전을 고정
    label_map = {word: idx for idx, word in enumerate(sorted(os.listdir(category_paths[0])))}
    print(f"라벨 매핑: {label_map}")

    # 4) 데이터 로드 : 서로 다른 전처리 버전의 동일 카테고리를 결합
    dataset_list = [NpySequenceDataset(path, label_map) for path in category_paths]
    dataset = ConcatDataset(dataset_list)
    if len(dataset) == 0:
        print(f"샘플이 없습니다: {category}, 스킵")
        continue

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5) 모델 초기화 : 입력/어텐션 차원과 클래스 수를 명시적으로 지정
    model = KeypointGRUModelV2(input_dim=152, attn_dim=146, num_classes=len(label_map)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 6) 학습 루프 : 순전파 → 손실 → 역전파 → 최적화 → 지표 집계
    for epoch in range(EPOCHS):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            _, preds = out.max(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            running_loss += loss.item() * y.size(0)

        epoch_loss = running_loss / total
        acc = correct / total * 100
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss : {epoch_loss:.4f} | Accuracy : {acc:.2f}%")

    # 7) 모델 및 라벨맵 저장 : 카테고리별 가중치와 레이블 매핑 보관
    model_path = os.path.join(MODEL_SAVE_DIR, f'{category}_model.pth')
    label_path = os.path.join(MODEL_SAVE_DIR, f'{category}_label_map.pkl')

    torch.save(model.state_dict(), model_path)
    with open(label_path, 'wb') as f:
        pickle.dump(label_map, f)

    print(f"[{category}] v2 혼합 모델 저장 완료 → {model_path}")
