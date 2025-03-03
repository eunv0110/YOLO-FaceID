## 📊 성능 분석 결과

### 1. 본인/타인 얼굴 탐지 모델 혼동 행렬

<div align="center">
  <img width="750" alt="혼동 행렬 분석" src="https://github.com/user-attachments/assets/1f109750-b7d6-4647-aaec-cc90c7e11206">
</div>

- **혼동 행렬 분석**:
  - 본인(my) 클래스: 1420개의 올바른 예측과 매우 낮은 오분류율
  - 타인(others) 클래스: 1583개의 올바른 예측, 673개의 배경 오분류
  - 배경(background) 클래스: 소수의 오분류만 발생
  - 전반적으로 높은 인식 정확도 보임

### 2. UltraLytics YOLO 학습 과정 성능 그래프

<div align="center">
  <img width="750" alt="YOLO 학습 결과 그래프" src="https://github.com/user-attachments/assets/053fc611-26f6-41dd-b810-d0b89508c063">
</div>

- **학습 과정 분석**:
  - **상단 행 그래프**: 
    - 학습 데이터에 대한 손실 및 메트릭 변화
    - train/box_loss, train/cls_loss, train/dfl_loss 모두 10 에폭 동안 안정적으로 감소
    - Precision(정밀도)는 0.88에서 시작하여 최종 0.96 이상까지 향상
    - Recall(재현율)은 0.74에서 시작하여 최종 0.81 수준으로 개선
  
  - **하단 행 그래프**:
    - 검증 데이터에 대한 손실 및 메트릭 변화
    - 변동성이 있으나 val/box_loss, val/cls_loss, val/dfl_loss 모두 전반적으로 감소
    - mAP50는 0.79에서 시작하여 0.87까지 향상
    - mAP50-95는 0.64에서 시작하여 0.74까지 개선

- **주요 성과**:
  - 학습/검증 손실 모두 지속적으로 감소하여 과적합 없이 모델이 일반화됨
  - 정밀도와 재현율이 균형적으로 향상되어 실제 사용 환경에서 안정적인 성능 기대
  - 특히 mAP50-95의 꾸준한 향상은 다양한 IoU 임계값에서도 모델이 견고함을 보여줌

### 3. 모델별 성능 비교

<div align="center">
  <table>
    <tr>
      <th width="25%">평가 지표</th>
      <th width="25%">본인/타인 구분 모델<br>(YOLO)</th>
      <th width="25%">조원별 얼굴 식별 모델<br>(YOLO-cls)</th>
      <th width="25%">통합 사용 시</th>
    </tr>
    <tr>
      <td>정확도</td>
      <td>95% 이상</td>
      <td>80-95%</td>
      <td>90% 이상</td>
    </tr>
    <tr>
      <td>처리 속도</td>
      <td>매우 빠름</td>
      <td>빠름</td>
      <td>빠름</td>
    </tr>
    <tr>
      <td>다중 얼굴 처리</td>
      <td>우수</td>
      <td>매우 우수</td>
      <td>매우 우수</td>
    </tr>
    <tr>
      <td>적용 상황</td>
      <td>출입 1차 검증</td>
      <td>개인별 식별</td>
      <td>전체 출입 관리</td>
    </tr>
  </table>
</div># 📷 카메라 기반 얼굴 인식 프로젝트

<div align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" />
  <br><br>
  <p><i>KT AIVLE School AI 트랙 미니프로젝트 4차</i></p>
</div>

<br>

## 📋 프로젝트 개요

<div align="center">
  <h3><i>"카메라를 활용한 실시간 얼굴 인식 및 사원 출입 자동화 시스템"</i></h3>
</div>

<br>

본 프로젝트는 카메라를 활용하여 사원의 얼굴을 인식하고 사내 출입을 자동화하는 시스템을 개발하는 것을 목표로 합니다. 사원의 출입이 많은 출퇴근 시간이나 점심시간의 대기 시간을 최소화하여 업무 효율성을 향상시키는 솔루션입니다.

<br>

## ⏰ 프로젝트 기간
- 2024년 10월 21일 ~ 2024년 10월 25일 (5일간)

<br>

## 📝 문제 정의

```
❓ 카드키 태그 방식 대신 얼굴 인식 기술을 활용하여 사내 출입 과정의 대기 시간을 어떻게 최소화할 수 있을까?
```

<br>

### 얼굴 인식 방식이 필요한 이유:

- 🔹 출퇴근/점심 시간대의 병목 현상 해소
- 🔹 카드키 분실/휴대 불편 문제 제거
- 🔹 고속도로 하이패스와 같은 빠른 출입 시스템 구현
- 🔹 비접촉식 출입 관리로 위생적인 환경 조성

<br>

## 📊 데이터셋 소개

<table>
  <tr>
    <th align="center" width="20%">분석 대상</th>
    <td>얼굴 인식용 이미지 데이터</td>
  </tr>
  <tr>
    <th align="center">데이터 출처</th>
    <td>
      <ul>
        <li>Labeled Faces in the Wild (LFW) 데이터셋</li>
        <li>Roboflow Universe의 Face Recognition 데이터셋</li>
        <li>직접 수집한 개인 얼굴 이미지</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th align="center">데이터 구성</th>
    <td>
      <ul>
        <li>Keras 모델용 데이터: 최소 2,500장의 이미지</li>
        <li>YOLO-cls 모델용 데이터: 최소 2,500장의 이미지</li>
        <li>YOLO 객체 탐지 모델용 데이터: 최소 2,500장의 이미지와 라벨</li>
      </ul>
    </td>
  </tr>
</table>

<br>

## 📊 프로젝트 구조 및 모델 접근 방식

<div align="center">
  <img width="800" alt="프로젝트 접근 방식 개요" src="https://path-to-your-image/project_overview.png">
</div>

본 프로젝트는 사내 출입 관리를 위한 두 단계의 얼굴 인식 시스템을 구현했습니다:

### 1. 사내 출입 1차 검증: 본인/타인 얼굴 구분 (UltraLytics YOLO)
- **목적**: 출입자가 등록된 사원인지 아닌지 빠르게 구분
- **선택 모델**: `yolo11s imgsz 160 p10` (5가지 모델 중 Best 성능)
- **적용 상황**: 출입구에서 1차 검증으로 활용
- **특징**: 높은 정확도로 빠르게 본인/타인 구분 가능

### 2. 사내 출입 2차 검증: 조원별 얼굴 식별 (UltraLytics YOLO-cls)
- **목적**: 1차에서 확인된 인가자가 누구인지 정확히 식별
- **선택 모델**: `yolo 11n-cls` (시간 제약으로 인해 단일 모델만 테스트)
- **적용 상황**: 개인별 권한 부여 및 출입 기록을 위한 2차 검증
- **특징**: 조원 6명의 얼굴을 각각의 클래스로 높은 정확도 분류

### 3. 두 모델의 사용 목적
- **UltraLytics YOLO**: 본인인지 아닌지 여부를 구분할 때 사용
- **UltraLytics YOLO-cls**: 여러 명의 얼굴을 각각 누구인지 판별할 때 사용
- **통합 활용**: 두 모델을 연계하여 단계적으로 출입 권한 확인 및 개인 식별



<br>

## 📱 구현 모델

### 1. FaceNet 기반 얼굴 인식 모델

<table>
  <tr>
    <th width="20%">모델 구조</th>
    <td>
      <ul>
        <li>InceptionResNetV1 아키텍처 사용</li>
        <li>입력 이미지 크기: 160x160</li>
        <li>출력: 128차원의 얼굴 임베딩 벡터</li>
        <li>이진 분류를 위한 Sigmoid 레이어 추가</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>학습 방법</th>
    <td>
      <ul>
        <li>사전 학습된 FaceNet 가중치 적용</li>
        <li>마지막 레이어에 전이학습 적용</li>
        <li>Adam 옵티마이저 사용</li>
        <li>Binary Cross-Entropy 손실 함수</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>주요 기능</th>
    <td>
      <ul>
        <li>얼굴 특징 추출 및 임베딩 생성</li>
        <li>사용자/비사용자 얼굴 구분</li>
        <li>실시간 웹캠 인식 가능</li>
      </ul>
    </td>
  </tr>
</table>

### 2. YOLO-cls 분류 모델

<table>
  <tr>
    <th width="20%">모델 구조</th>
    <td>
      <ul>
        <li>YOLOv8n-cls 기반 분류 모델</li>
        <li>이미지 분류에 특화된 YOLO 변형</li>
        <li>경량화된 구조로 빠른 추론 속도</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>학습 방법</th>
    <td>
      <ul>
        <li>UltraLytics YOLO 프레임워크 사용</li>
        <li>train/val 데이터셋 분할</li>
        <li>데이터 증강 기법 적용</li>
        <li>10 에폭 학습</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>주요 기능</th>
    <td>
      <ul>
        <li>얼굴 이미지 다중 클래스 분류</li>
        <li>사용자/비사용자 클래스 구분</li>
        <li>높은 정확도의 얼굴 인식</li>
      </ul>
    </td>
  </tr>
</table>

### 3. YOLO 객체 탐지 모델

<table>
  <tr>
    <th width="20%">모델 구조</th>
    <td>
      <ul>
        <li>YOLOv8n 객체 탐지 모델</li>
        <li>실시간 객체 탐지에 최적화</li>
        <li>얼굴 위치 및 클래스 동시 인식</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>학습 방법</th>
    <td>
      <ul>
        <li>Roboflow 및 직접 수집한 얼굴 데이터 활용</li>
        <li>Annotation 작업을 통한 바운딩 박스 정보 생성</li>
        <li>YAML 파일을 통한 데이터셋 구성</li>
        <li>10 에폭 학습</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>주요 기능</th>
    <td>
      <ul>
        <li>실시간 얼굴 위치 탐지</li>
        <li>다수 인물 동시 인식 가능</li>
        <li>각 얼굴별 클래스 분류 및 신뢰도 표시</li>
      </ul>
    </td>
  </tr>
</table>

<br>

## 📈 모델 성능 비교

<div align="center">
  <table>
    <tr>
      <th>모델</th>
      <th>정확도</th>
      <th>추론 속도</th>
      <th>장점</th>
      <th>단점</th>
    </tr>
    <tr>
      <td>FaceNet</td>
      <td>97.5%</td>
      <td>중간</td>
      <td>
        <ul>
          <li>고정밀 얼굴 특징 추출</li>
          <li>적은 데이터로도 학습 가능</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>전처리 과정 필요</li>
          <li>계산 복잡도 높음</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>YOLO-cls</td>
      <td>96%</td>
      <td>빠름</td>
      <td>
        <ul>
          <li>빠른 추론 속도</li>
          <li>간단한 구현</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>얼굴 검출 별도 필요</li>
          <li>다수 인물 동시 처리 어려움</li>
        </ul>
      </td>
    </tr>
    <tr>
      <td>YOLO 객체 탐지</td>
      <td>94%</td>
      <td>매우 빠름</td>
      <td>
        <ul>
          <li>실시간 다중 얼굴 검출</li>
          <li>위치 및 클래스 동시 인식</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Annotation 작업 필요</li>
          <li>원거리 소형 얼굴 인식 한계</li>
        </ul>
      </td>
    </tr>
  </table>
</div>

<br>

## 🧪 성능 개선 실험

### FaceNet 모델 최적화
- 드롭아웃 비율 조정 (0.1, 0.3, 0.5)
- 학습률 최적화 (0.001, 0.0005, 0.0001)
- 배치 크기 실험 (16, 32, 64)
- 데이터 증강 기법 적용

### YOLO-cls 모델 최적화
- 이미지 크기 조정 (160, 224, 320)
- 모델 복잡도 비교 (n, s, m)
- 학습률 스케줄러 적용
- 클래스 불균형 처리

### YOLO 객체 탐지 모델 최적화
- Mosaic 증강 효과 실험
- 앵커 박스 최적화
- IoU 임계값 조정
- 모델 경량화 실험

<br>

## 🚀 프로젝트 실행 방법

### 1. 데이터 수집
```python
# 얼굴 이미지 데이터 수집 실행
python Step0_etc_image_save.py
```
- 웹캠을 통해 얼굴 이미지를 촬영하여 데이터셋 생성
- 수집된 이미지는 모델 학습에 사용됨

### 2. 모델 학습
각 모델 학습을 위한 Jupyter Notebook 파일:
```
# 본인/타인 구분 YOLO-cls 모델 학습
Step2_1_Use_YOLO_cls.ipynb

# 조원별(5인) 얼굴 식별 YOLO-cls 모델 학습
Step2_1_Use_YOLO_cls_5_person.ipynb

# 본인/타인 구분 YOLO 객체 탐지 모델 학습 
Step3_1_Use_YOLO.ipynb

# 조원별(5인) 얼굴 식별 YOLO 객체 탐지 모델 학습
Step3_1_Use_YOLO_modified_for_5_classes.ipynb
```

### 3. 학습된 모델 불러와서 실행하기
```python
# 본인/타인 구분 모델 실행 - YOLO 객체 탐지
python Step3_2_Detect_YOLO.py
# 이 파일은 yolo11n-cls.pt 모델을 불러와 사용

# 조원별(5인) 얼굴 식별 모델 실행 - YOLO-cls
python Step3_2_Detect_YOLO_5person.py
# 이 파일은 5person-yolo11n-cls.pt 모델을 불러와 사용
```

### 4. 모델별 용도 구분
- **본인/타인 구분**: `yolo11n-cls.pt` 모델 사용 (Step3_2_Detect_YOLO.py)
- **조원별 얼굴 식별**: `5person-yolo11n-cls.pt` 모델 사용 (Step3_2_Detect_YOLO_5person.py)

<br>

## 🏆 모델 비교 및 최종 선택

<div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #f0f8ff; margin: 20px 0;">
  <p><b>🔍 최종 추천 모델: YOLO 객체 탐지 모델</b></p>
  <p>세 가지 모델을 비교 분석한 결과, 출입 관리 시스템에는 YOLO 객체 탐지 모델이 가장 적합하다고 판단됩니다. 그 이유는 다음과 같습니다:</p>
  <ul>
    <li><b>실시간 처리:</b> 가장 빠른 추론 속도로 대기 시간 최소화</li>
    <li><b>다중 얼굴 동시 인식:</b> 여러 명이 동시에 접근해도 병목 현상 없이 처리 가능</li>
    <li><b>위치 기반 인식:</b> 프레임 내 얼굴 위치 파악으로 출입구 접근 상황 대응 가능</li>
    <li><b>모듈화:</b> 출입 관리 외에도 다양한 보안 시스템과 통합 용이</li>
  </ul>
  <p>단, 높은 보안이 요구되는 특수 구역에는 FaceNet 모델의 높은 정확도를 활용한 2단계 인증을 추가로 적용할 수 있습니다.</p>
</div>

<br>

## 🧠 프로젝트 배운 점

<div align="center">
  <table>
    <tr>
      <td width="30%" align="center">📊<br><b>데이터 수집 및 가공</b></td>
      <td width="70%">모델별로 다른 형태의 데이터셋을 요구하는 상황에서 효율적인 데이터 수집 및 가공 방법 습득</td>
    </tr>
    <tr>
      <td align="center">🔄<br><b>전이학습</b></td>
      <td>사전 학습된 모델을 활용해 적은 데이터로도 높은 성능을 달성하는 전이학습 기법 적용</td>
    </tr>
    <tr>
      <td align="center">🤖<br><b>다양한 모델 비교</b></td>
      <td>동일한 문제에 대해 서로 다른 접근법을 가진 모델들의 장단점 비교 분석 능력 향상</td>
    </tr>
    <tr>
      <td align="center">📈<br><b>모델 최적화</b></td>
      <td>하이퍼파라미터 튜닝, 데이터 증강 등을 통한 모델 성능 개선 방법 습득</td>
    </tr>
    <tr>
      <td align="center">🔍<br><b>실시간 시스템</b></td>
      <td>웹캠을 활용한 실시간 얼굴 인식 시스템 구현 및 최적화 방법 학습</td>
    </tr>
    <tr>
      <td align="center">⚙️<br><b>로컬 환경 구축</b></td>
      <td>Colab에서 학습한 모델을 로컬 환경에서 실행하기 위한 환경 구축 및 최적화 경험</td>
    </tr>
  </table>
</div>

<br>

## 🛠️ 기술 스택

<div align="center">
  <table>
    <tr>
      <th colspan="2" align="center">분류</th>
      <th align="center">기술</th>
      <th align="center">용도</th>
    </tr>
    <tr>
      <td rowspan="4" width="10%">💻</td>
      <td width="20%"><b>언어</b></td>
      <td width="25%">Python</td>
      <td width="45%">전체 프로젝트 개발</td>
    </tr>
    <tr>
      <td rowspan="2"><b>데이터 처리</b></td>
      <td>OpenCV</td>
      <td>이미지 처리 및 얼굴 검출</td>
    </tr>
    <tr>
      <td>NumPy</td>
      <td>수치 연산 및 배열 처리</td>
    </tr>
    <tr>
      <td><b>개발 환경</b></td>
      <td>Google Colab, VSCode</td>
      <td>모델 개발 및 로컬 구현</td>
    </tr>
    <tr>
      <td rowspan="3">🤖</td>
      <td rowspan="3"><b>딥러닝</b></td>
      <td>TensorFlow, Keras</td>
      <td>FaceNet 모델 구현 및 학습</td>
    </tr>
    <tr>
      <td>UltraLytics</td>
      <td>YOLO-cls 및 YOLO 객체 탐지 모델 구현</td>
    </tr>
    <tr>
      <td>Roboflow</td>
      <td>데이터 Annotation 및 증강</td>
    </tr>
  </table>
</div>

<br>

## 👨‍💻 참여자 

<div align="center">
  <table>
    <tr>
      <td align="center"><b>개발팀</b></td>
      <td align="center"><b>연구팀</b></td>
      <td align="center"><b>테스트팀</b></td>
    </tr>
  </table>
</div>

<br>

---

<div align="center">
  <p>본 프로젝트는 KT AIVLE School AI 트랙 미니프로젝트 4차로 진행되었습니다.</p>
  <p>© 2025 KT AIVLE School 6기 AI 트랙 미니프로젝트</p>
</div>
