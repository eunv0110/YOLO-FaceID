import cv2
import numpy as np
from ultralytics import YOLO

###############################################
## 모델 불러오기
model = YOLO('C:/Users/User/Downloads/yolo11s-face.pt')

###############################################

## opencv에서 사용하려는 카메라
cap = cv2.VideoCapture(0)

## 카메라 동작 확인용
if not cap.isOpened():
    print('웹캠 실행 불가')
    exit()

## 사람들 이름 정의 (최대 5명까지)
person_names = ["My Face", "Person1", "Person2", "Person3", "Person4"]
# 각 사람별 색상 정의 (BGR 형식)
person_colors = [
    (0, 255, 0),    # 녹색 (My Face)
    (0, 165, 255),  # 주황색 (Person1)
    (0, 0, 255),    # 빨간색 (Person2)
    (255, 0, 0),    # 파란색 (Person3)
    (255, 0, 255)   # 자주색 (Person4)
]

## 클래스 ID 트래킹 위한 변수
last_person_id = 0  # 마지막으로 할당된 ID

## 얼굴 트래킹을 위한 변수들
face_trackers = {}  # 얼굴 추적 정보를 저장할 딕셔너리
next_face_id = 1    # 다음에 할당할 ID (0은 My Face용)

## 매 프레임마다 동작시킬 것이므로 무한 반복문
while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 로드 불가')
        break
    
    frame = frame.astype(np.uint8)
    ## 카메라 좌우 전환
    frame = cv2.flip(frame, 1)
    
    ## 예측값 생성
    results = model(frame, stream=True, conf=0.5, iou=0.3)
    
    ## 이번 프레임에서 감지된 얼굴들
    current_faces = []
    
    ## 예측한 것을 뜯어봅시다
    for r in results:
        ## r_b는 매 프레임의 바운딩 박스'들'의 정보를 가지고 있음
        r_b = r.boxes
        
        ## r_b의 클래스 예측값이 없는게 아니라면
        if not r_b.cls is None:
            ## r_b가 클래스 예측한만큼 반복 수행
            for idx in range(len(r_b)):
                ## 점 2개의 좌표를 가져옴
                x1, y1, x2, y2 = int(r_b.xyxy[idx][0]), int(r_b.xyxy[idx][1]), int(r_b.xyxy[idx][2]), int(r_b.xyxy[idx][3])
                
                ## 중심점 계산
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                ## 신뢰도 점수
                conf = r_b.conf[idx] * 100
                
                ## 원래 모델의 클래스 (0: My Face, 1: Other Face)
                original_class = int(r_b.cls[idx])
                
                ## 얼굴의 새 ID를 결정
                face_id = -1  # 초기값
                
                if original_class == 0:
                    # 내 얼굴이면 항상 ID 0
                    face_id = 0
                    person_idx = 0
                else:
                    # 다른 얼굴이면 위치 기반으로 ID 할당
                    # 이전에 추적 중이던 얼굴인지 확인
                    matched = False
                    
                    for tracker_id, tracker_info in face_trackers.items():
                        old_x, old_y = tracker_info['center']
                        # 이전 위치와 현재 위치 사이의 거리 계산
                        distance = np.sqrt((center_x - old_x)**2 + (center_y - old_y)**2)
                        
                        # 일정 거리 이내면 같은 사람으로 판단
                        if distance < 50 and tracker_id != 0:  # 거리 임계값은 조정 가능
                            face_id = tracker_id
                            matched = True
                            break
                    
                    # 새로운 얼굴이면 새 ID 할당
                    if not matched:
                        if next_face_id <= 4:  # 최대 5명(ID: 0-4)까지만 트래킹
                            face_id = next_face_id
                            next_face_id += 1
                        else:
                            # 이미 5명 꽉 찼으면 가장 오래된 ID를 재사용 (ID 0은 재사용 안함)
                            face_id = 1  # 임시로 1번부터 재사용
                
                # 얼굴 정보 업데이트
                if face_id >= 0 and face_id < len(person_names):
                    face_trackers[face_id] = {
                        'center': (center_x, center_y),
                        'last_seen': 0  # 프레임 카운터 (미사용)
                    }
                    
                    person_idx = face_id
                    color = person_colors[person_idx]
                    label_text = f'{person_names[person_idx]} conf: {conf:.2f}%'
                    
                    # 현재 프레임에서 감지된 얼굴 리스트에 추가
                    current_faces.append(face_id)
                    
                    ## 프레임에 박스 표시
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    ## 박스 위에 텍스트 추가
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2)
    
    # 현재 프레임에서 감지되지 않은 얼굴은 트래커에서 제거
    # (단순화를 위해 생략 - 필요시 구현)
    
    ## 프레임을 확인할 수 있다
    cv2.imshow('Face_Detection', frame)
    
    ## 키보드 q 키를 누르면 반복문 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()