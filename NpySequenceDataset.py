"""
Numpy 시퀀스(.npy) 로더용 PyTorch Dataset

목적 :
- 클래스별 폴더 구조로 저장된 시퀀스(.npy)를 순회하여 (시퀀스, 라벨)을 반환한다.

입력 형식 :
- root_dir/클래스명/*.npy 의 디렉터리 구조
- label_map : {클래스명: 정수라벨}
- 각 .npy는 (T, F) 형태(예: (90, 152))의 float 배열이어야 한다.

출력 형식 :
- __getitem__(i) → x: torch.FloatTensor(T, F), y: torch.LongTensor()
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NpySequenceDataset(Dataset):
    def __init__(self, root_dir, label_map):
        """
        파라미터 :
        - root_dir : 데이터 루트 디렉터리(클래스별 하위 폴더 포함)
        - label_map : 폴더명(클래스명) → 정수 라벨 매핑
        """
        self.samples = []
        self.label_map = label_map

        for word in os.listdir(root_dir):
            word_path = os.path.join(root_dir, word)
            if not os.path.isdir(word_path):
                continue
            label = label_map.get(word)
            if label is None:
                continue
            for file in os.listdir(word_path):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(word_path, file), label))

        print(f"[INFO] 총 {len(self.samples)}개 샘플 로드됨 - from: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path)  # (T, F) 예: (90, 152)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
