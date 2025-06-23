# Attention Analysis Tool

언어 모델의 어텐션 메커니즘을 분석하고 시각화하는 도구입니다. 다양한 도메인의 데이터셋에 대해 모델의 어텐션 패턴을 분석하고, evidence 토큰에 대한 반응을 시각화합니다.

## 🚀 주요 기능

### 1. 모델 관리
- Huggingface 모델 로드/해제
- 모델 상태 확인
- 새로고침 시에도 모델 유지

### 2. 데이터셋 생성
- 다양한 도메인(Medical, Legal, Technical, General) 지원
- 도메인별 프롬프트 자동 생성
- Evidence 토큰 자동 추출

### 3. 어텐션 분석
- 프롬프트별 어텐션 패턴 시각화
- Evidence 토큰에 대한 어텐션 분석
- 헤드별 어텐션 히트맵 생성

### 4. 일괄 실험
- 다중 도메인 실험 지원
- 최대 10,000개 프롬프트 처리 가능
- 실험 결과 자동 저장

## 🛠️ 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd attention-analysis-tool
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. Ollama 설치 (필수)
- [Ollama 공식 사이트](https://ollama.ai/)에서 설치
- 지원 모델: llama2, mistral, gemma 등

## 🎮 사용 방법

1. 앱 실행
```bash
streamlit run app.py
```

2. 모델 로드
- "모델 로드" 탭에서 사용할 모델 선택
- 모델 로드 버튼 클릭

3. 데이터셋 생성
- "데이터셋 생성" 탭에서 도메인 선택
- 생성할 데이터셋 수 지정
- 생성 버튼 클릭

4. 실험 실행
- "실험" 탭에서 도메인과 데이터셋 선택
- 프롬프트 선택 및 어텐션 분석
- 일괄 실험 실행 가능

5. 결과 확인
- "실험 기록" 탭에서 저장된 실험 결과 확인
- 도메인별 어텐션 패턴 분석
- 히트맵 및 통계 확인

## 📊 실험 결과

실험 결과는 다음과 같은 정보를 포함합니다:
- 도메인별 evidence 반응 헤드 분포
- 헤드-토큰 어텐션 히트맵
- 토큰-헤드 어텐션 히트맵
- 메타데이터 (도메인, 모델, 타임스탬프 등)

## 🔧 기술 스택

- Python 3.10+
- Streamlit
- PyTorch
- Transformers
- Ollama
- Pandas
- Matplotlib
- Seaborn

## 📝 라이선스

MIT License

## 👥 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# Hugging Face 토큰(.env) 설정 방법

1. [Hugging Face 토큰 발급](https://huggingface.co/settings/tokens)
   - "New token" 클릭
   - 이름 입력, 권한은 **Read**로 설정
   - 토큰 복사 (hf_로 시작)

2. 프로젝트 루트에 `.env` 파일 생성 후 아래와 같이 입력

```
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

3. python-dotenv가 자동으로 .env 파일을 읽어 환경변수로 등록합니다.

4. Llama-2 등 gated 모델을 사용할 경우, Hugging Face에서 모델 접근 승인을 받아야 합니다. 