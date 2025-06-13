# ConvNeXt 학습 가이드

이 문서는 ConvNeXt 모델을 학습시키는 Python 스크립트(`train.py`)와 사용 예시입니다.

---

[Uploading data_to_csv.py…]()


## 1. 데이터 분할 및 CSV 생성 스크립트 사용법

`data_to_csv.py` 스크립트는 원본 이미지 폴더를 train/val로 5:5 비율로 분할하고, 각 폴더별로 레이블이 포함된 CSV 파일을 자동 생성해 줍니다.

```bash
python data_to_csv.py \
  --original_dir "/home/kkh/과업 1-B 상황별(놀이별) 사진분류/활동분류/데이터 폴더" \
  --train_csv "/home/kkh/과업 1-B 상황별(놀이별) 사진분류/활동분류/image_labels_train.csv" \
  --val_csv   "/home/kkh/과업 1-B 상황별(놀이별) 사진분류/활동분류/image_labels_val.csv" \
  --val_ratio 0.2
```

* `--original_dir` (`-o`): 원본 이미지가 들어있는 최상위 폴더 경로 (위치가 정확해야 합니다. copy path를 이용하시는 것을 추천드립니다.)
* `--train_csv` (`-t`): 생성할 학습용 CSV 파일 경로
* `--val_csv` (`-v`): 생성할 검증용 CSV 파일 경로
* `--val_ratio` (`-r`): 검증 데이터로 사용할 비율을 0보다 크고 1보다 작은 실수로 지정합니다. 예를 들어 0.2는 전체의 20%를 val 폴더로 분할합니다.

* 참고

  
  ![image](https://github.com/user-attachments/assets/48badbb3-29ba-452f-9ccd-045e4def051f)

  
위 사진과 같은 형태로 데이터 폴더가 구성되어있어야 하며 각 하위폴더(e.g. 2-5.수과학탐구)에는 해당 클래스에 해당하는 이미지들이 있어야합니다.


실행 후, 다음과 같은 작업이 수행됩니다:

1. `train` 및 `val` 폴더가 자동 생성됩니다.
2. 각 클래스(서브폴더)별로 이미지 파일을 7:3 비율로 분할하여 해당 폴더로 이동합니다.
3. 이동된 이미지 경로와 클래스 정보로 CSV 파일을 생성합니다。

**이제 생성된 `image_labels_train.csv`와 `image_labels_val.csv`를 `train.py`의 `--train-csv` 및 `--val-csv` 인자로 사용하시면 됩니다。**

---

## 2. 준비사항

* Python 3.8 이상
* PyTorch
* torchvision
* pandas
* numpy
* Pillow

```bash
pip install torch torchvision pandas numpy pillow
```

## 3. 코드 개요

* **train.py**: ConvNeXt 모델을 불러와 Mixup/Cutmix을 적용해 학습하는 스크립트

  * `ImageDataset` 클래스: CSV로부터 이미지 경로와 레이블을 로드
  * `mixup_data` / `cutmix_data` 함수: Mixup 및 Cutmix 증강
  * `train_model` 함수: 학습 루프, Early Stopping, 최저 손실 모델 저장
  * `evaluate_model` 함수: 검증 정확도 계산 및 결과 CSV 저장

## 4. 디렉토리 구조 예시

```
project-root/
├── train.py                   # 학습 스크립트
├── image_labels_train.csv     # 학습용 레이블 CSV
├── image_labels_val.csv       # 검증용 레이블 CSV
├── data_to_csv.py             # 데이터 분할 및 CSV 생성 스크립트
└── logs/                      # 학습 결과 및 체크포인트 저장 폴더
```

## 5. ConvNeXt 학습 실행 방법

터미널(또는 명령 프롬프트)에서 `train.py`를 실행합니다。

```bash
python train.py \
  --model convnext_tiny \
  --train-csv /path/to/image_labels_train.csv \
  --val-csv /path/to/image_labels_val.csv \
  --batch-size 128 \
  --epochs 90 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --output-dir ./logs/convnext_base \
  --onnx-name convnext_tiny.onnx
```

* `--model`: 사용할 ConvNeXt 모델 종류 (`convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`)
* `--train-csv`: 학습용 CSV 파일 경로 (data_to_csv.py를 실행하고 나온 image_labels_train.csv를 사용하면 됩니다.)
* `--val-csv`: 검증용 CSV 파일 경로 (data_to_csv.py를 실행하고 나온 image_labels_val.csv를 사용하면 됩니다.)
* `--batch-size`: 배치 크기 (메모리 상황에 맞게 조정)
* `--epochs`: 총 학습 에폭 수
* `--lr`: 초기 학습률
* `--weight-decay`: 가중치 감쇠(정규화) 값
* `--output-dir`: 모델 체크포인트와 로그 저장 폴더
* `--onnx-name`: ONNX로 export할 파일명

## 6. 학습 결과

* 학습 도중 가장 낮은 검증 손실을 기록한 모델이 `logs/convnext_base/best_{model}_epoch*.pth` 형태로 저장됩니다。
* 학습이 끝나면 검증 정확도가 출력되고, `evaluation_results.csv` 파일에 예측 결과와 정답이 함께 저장됩니다。

## 7. 팁

* **Early Stopping**: `--patience` 옵션(기본 20)으로 손실이 개선되지 않아도 멈추기 전 기다릴 에폭 수 지정
* **Mixup/Cutmix 강도**: `--alpha` 값을 조정해 증강 강도 변경 가능
* **Cutmix 확률**: `--cutmix-prob`로 Mixup 대신 Cutmix를 적용할 확률 설정
