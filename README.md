---
title: ASL Recognition
sdk: docker
app_port: 7860
pinned: false
---

# ASL Recognition - MediaPipe + XGBoost

Project nhan dien ky hieu ASL gom `A-Z`, `Space`, `Nothing`. Ban web dung webcam cua trinh duyet va gui tung frame ve backend Flask de du doan.

## Chay Web App Local

```cmd
py -m pip install -r requirements.txt
py app.py
```

Mo trinh duyet tai, bam `Start Camera`, roi cho phep quyen camera:

```text
http://127.0.0.1:5000
```

API du doan anh:

```cmd
curl -X POST -F "image=@test/A/A_0002.jpg" http://127.0.0.1:5000/api/predict
```

## Deploy Len Hugging Face Spaces

1. Tao tai khoan hoac dang nhap Hugging Face.
2. Vao `https://huggingface.co/new-space`.
3. Dat ten Space, chon `Docker` o muc SDK, roi tao Space.
4. Clone Space ve may:

```cmd
git clone https://huggingface.co/spaces/<username>/<space-name>
cd <space-name>
```

5. Copy cac file sau tu project nay vao thu muc Space:

```text
app.py
asl_model.py
model_baseline.pkl
hand_landmarker.task
requirements.txt
Dockerfile
.dockerignore
README.md
templates/
static/
```

6. Commit va push:

```cmd
git add .
git commit -m "Deploy ASL recognition app"
git push
```

Sau khi push, Hugging Face se tu build Docker image. Khi build xong, app chay truc tiep tren URL cua Space.

Luu y: webcam tren Hugging Face chay bang camera cua trinh duyet nguoi dung. Khong dung `cv2.VideoCapture(0)` tren server, vi server Hugging Face khong co camera cua may ban.

## Deploy Bang Docker Local

Build image:

```cmd
docker build -t asl-recognition .
```

Run container:

```cmd
docker run --rm -p 5000:7860 asl-recognition
```

Sau do mo:

```text
http://127.0.0.1:5000
```

## Deploy Len Render/Railway/Server

Lenh start tren Windows:

```cmd
gunicorn --bind 0.0.0.0:%PORT% app:app
```

Neu dich vu dung bien moi truong Unix:

```sh
gunicorn --bind 0.0.0.0:$PORT app:app
```

File can co khi deploy:

```text
app.py
asl_model.py
model_baseline.pkl
hand_landmarker.task
requirements.txt
templates/
static/
```

Co the dat duong dan model bang bien moi truong:

```text
ASL_MODEL_PATH=/path/to/model_baseline.pkl
ASL_HAND_TASK_PATH=/path/to/hand_landmarker.task
```

## Chay Du Doan Anh Don Bang CLI

```cmd
py asl_model.py --image "test\A\A_0002.jpg" --model model_baseline.pkl --hand-task hand_landmarker.task
```

Ket qua co dang:

```text
{"label": "A", "confidence": 0.99}
```

## Demo Webcam Local

```cmd
py asl_model.py --webcam --model model_baseline.pkl --hand-task hand_landmarker.task
```

Nhan `q` de thoat.

## Demo Webcam Thanh Chu Tieng Viet

Chay nhan dien local va ghep chu theo Telex:

```cmd
py vietnamese_webcam.py --model model_baseline.pkl --hand-task hand_landmarker.task
```

Cach ghep:

```text
O + O -> ô
O + X -> õ
O + O + X -> ỗ
A + W + S -> ắ
D + D -> đ
```

Trong cua so webcam:

```text
q: thoat
b: xoa 1 ky tu
c: xoa het
n: xuong dong
```

Neu muon xuat chu in hoa:

```cmd
py vietnamese_webcam.py --uppercase-output
```

## Chay Tren Google Colab

Upload cac file:

```text
asl_model.py
model_baseline.pkl
hand_landmarker.task
requirements.txt
```

Cai thu vien:

```python
!pip install -r requirements.txt
```

Predict anh:

```python
from asl_model import ASLRecognizer

recognizer = ASLRecognizer(
    model_path="model_baseline.pkl",
    hand_task_path="hand_landmarker.task",
)

label, confidence = recognizer.predict_image("A_0002.jpg")
print("Label:", label)
print("Confidence:", confidence)
```

Luu y: Google Colab khong chay webcam realtime bang `cv2.VideoCapture(0)` nhu may local. Neu can webcam tren Colab thi chup anh qua JavaScript roi dua anh do vao `predict_image`.
