# 📷 카메라 기반 얼굴 인식 프로젝트

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

본 프로젝트는 얼굴 인식을 위한 계층적 접근 방식을 채택했습니다:

### 1. 1단계: 본인/타인 얼굴 구분 (UltraLytics YOLO)
- **목적**: 얼굴이 본인(인가된 사용자)인지 타인인지 구분
- **데이터 구성**: my_face와 other_face로 이진 분류
- **특징**: 출입 권한 부여를 위한 첫 단계 인증 시스템

### 2. 2단계: 개인별 얼굴 식별 (UltraLytics YOLO-cls)
- **목적**: 6명의 조원 중 누구인지 정확히 식별
- **데이터 구성**: 조원 6명의 데이터를 각각의 클래스로 분류
- **특징**: 인가된 사용자들 간의 세부 식별 시스템

### 3. 데이터 처리 과정
1. **데이터 수집**: 각 조원별 얼굴 이미지 수집
2. **데이터 증강**: Annotation 및 Augmentation 기법 적용
3. **모델 학습**: 각 단계별 모델 구축 및 학습
4. **성능 평가**: 실시간 웹캠 테스트를 통한 성능 검증



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

### 1. 환경 설정
```bash
# 1. 프로젝트 폴더 생성 및 이동
cd Desktop
mkdir project4
cd project4

# 2. 가상환경 생성 및 활성화
python -m venv proj4
cd proj4/Scripts
activate

# 3. 필요 라이브러리 설치
pip install -r requirements.txt
```

### 2. 모델 실행 (로컬 환경)

#### FaceNet 모델 실행
```python
# 저장된 모델 로드
python Step1_2_Detect_Keras_aivler.py
```

#### YOLO-cls 모델 실행
```python
# 저장된 모델 로드
python Step2_2_Detect_YOLOcls_aivler.py
```

#### YOLO 객체 탐지 모델 실행
```python
# 저장된 모델 로드
python Step3_2_Detect_YOLO_aivler.py
```

### 3. 멀티 얼굴 인식 테스트
```python
# 다중 얼굴 인식 테스트
python multi_face_test.py
```

## 📋 일별 미션 수행 과정

<table>
  <tr>
    <th width="20%">날짜</th>
    <th width="80%">미션 내용 및 수행 결과</th>
  </tr>
  <tr>
    <td><b>1일차</b></td>
    <td>
      <ul>
        <li><b>FaceNet 모델 구현</b>
          <ul>
            <li>STEP 1: 본인/타인 얼굴 이미지 데이터셋 수집 및 로드</li>
            <li>STEP 2: 데이터 전처리 및 FaceNet 모델 구조 생성 (160×160 입력, 128차원 출력)</li>
            <li>STEP 3: 다양한 모델 실험, 모델 저장(.keras), 로컬 환경 웹캠 테스트</li>
          </ul>
        </li>
        <li>이슈 및 해결: 데이터 불균형 문제 → 데이터 증강 및 클래스 가중치 적용</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><b>2일차</b></td>
    <td>
      <ul>
        <li><b>YOLO-cls 모델 구현</b>
          <ul>
            <li>STEP 1-2: 데이터셋 로드 및 YOLO-cls 요구 폴더 구조로 전처리</li>
            <li>STEP 3: UltraLytics YOLO-cls 모델 선택 및 학습, 추론, 모델 저장(.pt), 로컬 테스트</li>
          </ul>
        </li>
        <li>이슈 및 해결: 모델 크기와 성능 균형 → YOLOv8n-cls 선택으로 속도와 정확도 최적화</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><b>3일차</b></td>
    <td>
      <ul>
        <li><b>Annotation 작업</b>
          <ul>
            <li>STEP 1: 본인 얼굴 이미지 수집 및 Annotation 작업 (Roboflow 활용)</li>
            <li>데이터 증강(Augmentation) 작업을 통한 데이터셋 확장</li>
          </ul>
        </li>
        <li>이슈 및 해결: Annotation 작업의 정확성 → 온라인 도구 활용으로 효율성 증대</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><b>4일차</b></td>
    <td>
      <ul>
        <li><b>YOLO 객체 탐지 모델 구현</b>
          <ul>
            <li>STEP 2: 데이터셋 로드 및 YOLO 요구 구조로 전처리, YAML 파일 생성</li>
            <li>STEP 3: UltraLytics YOLO 모델 선택, 학습, 추론, 모델 저장, 로컬 테스트</li>
          </ul>
        </li>
        <li>이슈 및 해결: 다중 얼굴 동시 인식 → 적절한 IoU 임계값 조정으로 개선</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><b>5일차</b></td>
    <td>
      <ul>
        <li><b>모델 최적화 및 비교 분석</b>
          <ul>
            <li>세 가지 모델(Keras, YOLO-cls, YOLO) 모두 하이퍼파라미터 튜닝으로 개선</li>
            <li>로컬 웹캠에서 실제 성능 테스트 및 지표화</li>
            <li>다중 인원 동시 인식 상황에서 각 모델별 성능 차이 분석</li>
            <li>결과 정리 및 발표 자료 준비</li>
          </ul>
        </li>
        <li>최종 결론: YOLO 객체 탐지 모델이 실시간 다중 얼굴 인식에 가장 적합</li>
      </ul>
    </td>
  </tr>
</table>

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
      <td align="center"><b>김수란</b></td>
      <td align="center"><b>김예은</b></td>
      <td align="center"><b>김태헌</b></td>
      <td align="center"><b>윤종진</b></td>
      <td align="center"><b>정요한</b></td>
      <td align="center"><b>황은비</b></td>
    </tr>
  </table>
</div>

<br>

---

<div align="center">
  <p>본 프로젝트는 KT AIVLE School AI 트랙 미니프로젝트 4차로 진행되었습니다.</p>
  <p>© 2025 KT AIVLE School 6기 AI 트랙 미니프로젝트</p>
</div>
