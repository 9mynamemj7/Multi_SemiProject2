# Image Classification & Colorization

### 🐤프로젝트 주제

    - 포켓몬스터 캐릭터 이미지 분류 (CNN)
    - 손그림 이미지 분류 후 간단 채색 (colorization)
    
### 🐤진행과정
    1. 캐릭터별 이미지 크롤링 (8가지 캐릭터)
    2. 이미지 전처리와 라벨링 (불량 데이터 정리, resize, 이미지별 라벨 입력)
    3. CNN 모델 생성
    4. 채색 모델 생성
    5. 모델 검증 및 확인
    
    [ 이미지 전처리 유형 ]
    1) resize 이후 그대로 모델링           (성진)
    2) resize 이후 이미지 노이즈 제거      (수진)
        - GaussianBlur와 NLmeans 비교
        - cv2.getStructuringElement와 morphologyEx를 사용해 노이즈 제거 
    4) resize 이후 이미지의 선만 따서 학습 (석찬)
    5) resize 이후 2차원으로 변경하여 학습 (민중)
    6) resize 이후 노이즈 제거, 선을 따서 학습
    7) resize 이후 노이즈 제거, 2차원으로 변경해 학습
        위의 유형들을 통해 결과 비교해보기
