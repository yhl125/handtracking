# Hand Tracking 코드 활용법

## 1. create_dataset.py
- 6번째 line의 actions 배열에는 학습시킬 모션 데이터 이름을 입력.
- 8번째 line의 secs_for_action에는 웹캠을 통해 학습시킬 동작의 녹화 시간 설정
- 코드를 실행시키면 웹캠이 켜지고, 입력한 순서대로 action을 녹화하면 dataset 폴더에 학습 데이터가 생성됨.

## 2. train.ipynb
- 마찬가지로 actions 배열에는 학습시킬 모션 데이터 이름을 입력.
- data 변수에는 로드할 학습 데이터를 입력.
- 이후 순서대로 코드를 진행시키면 models 폴더에 test.h5파일이 생성됨. (이 파일이 학습 모델임)

## 3. unity_python.py / unity.cs
- 유니티 환경에서 우선 unity.cs 스크립트를 실행시켜주면 로컬 웹소켓 서버가 열림.
- 이후 파이썬 환경에서 unity_python.py 코드를 실행시켜주면 웹소켓 클라이언트가 로컬 웹소켓 서버에 접속.
- 파이썬 코드가 실행된 뒤 웹캠이 열리는데, 이후 전송할 동작을 웹캠에 보여주면 해당 액션을 파이썬에서 인식하고 action이름을 표시해 줌. 
- 의도한 모션이 맞으면 q를 눌러서 유니티에 모션을 string 형태로 전송할 수 있음.

### full code uri https://github.com/yhl125/cardgame-unity/tree/feature/hand_tracking

pip install -r requirements.txt