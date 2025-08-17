"""
키포인트 기반 GRU 분류 모델(v2)

목적 :
- 시계열 키포인트 입력을 GRU로 인코딩하고, 손 관련 특징에 대해 어텐션을 적용한 뒤
  최종 클래스를 예측한다.

입력 형식 :
- x : (B, T, input_dim). 기본값 input_dim=152 (예 : 손/팔/어깨 등 전체 피처)
- 어텐션용 피처 길이 attn_dim=146 (예 : 손 좌표 + 손가락 각도 합산)

출력 형식 :
- (B, num_classes)

주의 :
- 어텐션은 입력 `x`의 앞쪽 `attn_dim` 차원만 사용한다. 따라서 `input_dim >= attn_dim` 조건이
  충족되어야 하며, 슬라이싱 규칙이 데이터 생성 파이프라인과 일치해야 한다.
"""

import torch
import torch.nn as nn

class KeypointGRUModelV2(nn.Module):
    def __init__(self, input_dim=152, attn_dim=146, hidden_dim=256, num_classes=6):
        """
        하이퍼파라미터 :
        - input_dim : 전체 피처 차원 (기본 152)
        - attn_dim : 어텐션에 사용할 피처 차원 (예 : 손 좌표 + 손가락 각도 146)
        - hidden_dim : GRU hidden size
        - num_classes : 클래스 개수
        """
        super().__init__()
        # 시계열 인코더 : (B, T, input_dim) → (B, T, hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # 어텐션 레이어 : (B, T, attn_dim) → (B, T, 1)
        self.attn_proj = nn.Sequential(
            nn.Linear(attn_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 최종 분류기 : (B, hidden_dim) → (B, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (B, T, 152)
        # 1) GRU 인코딩
        rnn_out, _ = self.gru(x)  # (B, T, H)

        # 2) 어텐션은 손 관련 피처에 대해서만 계산
        #    예 : 손 좌표(126) + 손가락 각도(20) = 146
        hand_feat = x[:, :, :146]  # (B, T, 146)
        attn_weights = torch.softmax(self.attn_proj(hand_feat), dim=1)  # (B, T, 1)

        # 3) GRU 출력 가중합(시간 축에 대해 합산)
        feat = (rnn_out * attn_weights).sum(dim=1)  # (B, H)

        # 4) 최종 클래스 예측
        return self.classifier(feat)  # (B, num_classes)
