# OpenSign — 단어 인식 모델 소개 (GRU + Attention)

웹캠으로부터 전신·얼굴·양손 키포인트를 추출해 **시퀀스(길이 T)**로 구성하고, GRU 기반 인코더 + 손 관련 피처에 대한 시간적 어텐션으로 단어(카테고리 내 단어 클래스)를 예측합니다. 실시간 데모에서는 수집(s) → 예측 → 신뢰도 기준 조건부 저장까지 한 번에 수행합니다. 

## 1) 모델 구성(Model)

- 입력: x ∈ ℝ^{B×T×F}, 기본 F = 152 (양손 상대좌표(126) + 손가락 관절각(20) + 손–얼굴 관계(6))

- 인코더: GRU(input_dim=152, hidden_dim=256) — 시계열을 은닉표현으로 변환

- 어텐션(손 피처 전용): 입력의 앞쪽 146차원(손 좌표·각도)만 사용해 attn_weights = softmax(MLP(146)) 계산 → Σ_t( attn_t · rnn_out_t )로 시간축 가중합

- 분류기: Linear(256→128) + ReLU + Dropout(0.3) + Linear(128→num_classes)

- 출력: logits ∈ ℝ^{B×num_classes} (카테고리별 단어 분류)


주의: 데이터 파이프라인이 피처 슬라이싱 규칙(앞 146차원이 손 관련 피처)에 맞게 생성되어야 합니다. input_dim ≥ attn_dim 조건 충족 필수. 

---

## 2) 특징 설계(Feature Engineering)

- 실시간 추론기에서 MediaPipe Holistic으로 랜드마크를 얻고 아래 피처를 생성합니다.

- 양손 상대좌표(126): 각 손의 21×(x,y,z)를 손목 기준 상대좌표로 정규화 (위치 불변성)

- 손가락 관절 각도(20): 엄지~새끼까지 연속 3점 각도(라디안)

- 손–얼굴 관계(6): 좌·우 손목과 코(대리점) 간 상대벡터
→ 총 152차원을 프레임당 1개 벡터로 만들고, 시퀀스 길이 T=FRAME_TARGET(기본 90)으로 누적합니다. 

---

## 3) 학습 파이프라인(Training)

- 데이터 구조: BASE_DIRS(예: data/) 아래 카테고리/단어/시퀀스.npy 구조, 각 .npy는 (T,F)(예: (90, 152)) 형상. 동일 카테고리를 여러 전처리 버전에서 불러와 ConcatDataset으로 결합(일반화↑). 

- 라벨 매핑: 첫 번째 데이터셋의 클래스 폴더명으로 label_map 고정(단어→index). 

- 하이퍼파라미터: batch_size=4, epochs=20, lr=1e-4(Adam) / 디바이스는 cuda 우선. 

- 모델 초기화: KeypointGRUModelV2(input_dim=152, attn_dim=146, num_classes=len(label_map))
→ CrossEntropy로 학습, 에폭별 Loss/Acc 로깅. 

- 산출물(카테고리별):

1. models_by_category_mixed_v2/{category}_model.pth (가중치)

2. models_by_category_mixed_v2/{category}_label_map.pkl (라벨 맵) 

요지: 카테고리별 독립 모델을 학습해, 같은 카테고리 안에서 단어 클래스를 분류합니다. (데이터가 늘수록 카테고리 확장/합본도 가능) 

---

## 4) 실시간 추론(Real-time)

- 입력/수집: VideoCapture(0)로 프레임 획득 → Holistic으로 키포인트 → 각 프레임 152차원 특징 추출 → **deque(maxlen=90)**에 누적. 수집 상태는 s 키로 토글. 

- 모델 로드: 실행 시 카테고리명을 입력하면 해당 카테고리의 *_model.pth와 *_label_map.pkl을 로드해 예측. 

- 예측/표시: 길이 90 시퀀스가 채워지면 softmax로 신뢰도 계산 → 텍스트 오버레이로 (라벨, 확률%) 표시(한글 폰트 사용). 

- 조건부 저장(옵션): CONF_THRESHOLD=90% 이상이면 collected_data/{카테고리}/{라벨}/{구간}/timestamp.npy에 시퀀스 저장. 구간은 90_95, 95_100, 100으로 분기. 종료는 q. 

---

## 5) 빠른 시작(실행 예시)

학습
```text
# 1) 의존성 설치 (가상환경 권장)
pip install -r requirements.txt

# 2) 데이터 폴더 준비
# data/feature_data_90fps/<category>/<word>/*.npy  (T=90, F=152)
# data/feature_data_90fps_v1/<category>/<word>/*.npy  (전처리 변형 버전)

# 3) 학습 실행 (카테고리별 혼합 학습 & 저장)
python train_v2.py
# -> models_by_category_mixed_v2/<category>_model.pth
#    models_by_category_mixed_v2/<category>_label_map.pkl
```

실시간 추론/수집

```text
python real_time_v2.py
# "테스트할 카테고리 이름" 입력 → 모델/라벨맵 로드
# 's' : 90프레임 수집 시작 → 예측/표시 → (신뢰도≥90%) 저장
# 'q' : 종료
```

## 6) 디렉터리 구조(예시)

```text
OpenSign-Word/
├─ data/
│  └─ <category>/<word>/*.npy        # (T=90, F=152)
├─ models_by_category_mixed_v2/
│  ├─ <category>_model.pth
│  └─ <category>_label_map.pkl
├─ train_v2.py                           # 카테고리별 혼합 학습 스크립트
├─ model_v2.py                           # GRU + Attention 모듈
└─ real_time_v2.py                       # 실시간 예측/조건부 저장 데모
```
