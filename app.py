import streamlit as st
import os

# 페이지 설정
st.set_page_config(
    page_title="Attention Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 앱 초기화 - 한 번만 실행
@st.cache_resource
def initialize_app():
    """앱 초기화 및 모듈 로드"""
    # 모듈들을 미리 로드하여 캐시
    from tabs import model_loader, dataset_generator, evidence_extractor, experiment, experiment_log, monitoring
    return {
        'model_loader': model_loader,
        'dataset_generator': dataset_generator,
        'evidence_extractor': evidence_extractor,
        'experiment': experiment,
        'experiment_log': experiment_log,
        'monitoring': monitoring
    }

# 앱 모듈들 초기화
if 'app_modules' not in st.session_state:
    st.session_state.app_modules = initialize_app()

# 사이드바 네비게이션
st.sidebar.title("🧠 Navigation")

# 페이지 선택
page = st.sidebar.selectbox(
    "페이지 선택",
    ["🏠 홈", "🤖 모델 로드", "📝 도메인 프롬프트 생성", "🔍 Evidence 추출", "🔬 실험", "📊 실험기록", "📈 모니터링"],
    index=0
)

# 메인 콘텐츠 영역
st.title("🧠 Attention Analysis")

# 페이지별 콘텐츠 표시
if page == "🏠 홈":
    # 앱 소개
    st.markdown("""
    ## 📋 앱 개요

    이 애플리케이션은 언어 모델의 어텐션 패턴을 분석하여 evidence 토큰에 대한 헤드별 반응을 연구합니다.

    ### 🎯 주요 기능

    1. **🤖 모델 로드**: 다양한 언어 모델을 로드하고 관리
    2. **📚 데이터셋 생성**: 도메인별 프롬프트 데이터셋 생성
    3. **🔬 실험**: 어텐션 패턴 실험 수행
    4. **📊 실험기록**: 실험 결과 분석 및 시각화
    5. **📈 모니터링**: 실시간 모델 성능 모니터링

    ### 🔬 연구 목적

    - **Evidence 토큰 식별**: 각 도메인에서 핵심적인 토큰들을 자동으로 식별
    - **어텐션 패턴 분석**: 모델의 각 어텐션 헤드가 evidence 토큰에 어떻게 반응하는지 분석
    - **도메인별 특성**: Medical, Legal, Technical, General 도메인별 어텐션 패턴 차이 연구
    - **해석 가능성**: 모델 내부 동작의 해석 가능성 향상

    ### 📊 지원 도메인

    - **Medical**: 의학 관련 프롬프트 및 전문 용어
    - **Legal**: 법률 관련 프롬프트 및 법적 개념
    - **Technical**: 기술 관련 프롬프트 및 프로그래밍 개념
    - **General**: 일반적인 프롬프트 및 범용 개념

    ### 🚀 시작하기

    왼쪽 사이드바에서 원하는 기능을 선택하여 시작하세요.

    ---

    ### 📈 최근 업데이트

    - **Evidence 추출 개선**: LLM 기반 토큰 추출 + 코드 기반 인덱스 계산
    - **사이드바 네비게이션**: 깔끔한 사이드바 기반 페이지 전환
    - **성능 최적화**: 모델 캐싱 및 메모리 관리 개선
    """)

    # 현재 상태 표시
    st.markdown("### 🔍 현재 상태")

    # 모델 로드 상태 확인
    if 'model' in st.session_state and st.session_state['model'] is not None:
        model_name = st.session_state.get('model_name', '알 수 없는 모델')
        st.success(f"✅ 모델 로드됨: {model_name}")
    else:
        st.warning("⚠️ 모델이 로드되지 않음")

    # 데이터셋 상태 확인
    dataset_path = "dataset"
    if os.path.exists(dataset_path):
        dataset_count = len([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
        st.info(f"📁 데이터셋 디렉토리: {dataset_count}개 모델 데이터셋")
    else:
        st.warning("⚠️ 데이터셋 디렉토리가 없음")

    # 실험 결과 상태 확인
    experiment_path = "experiment_results"
    if os.path.exists(experiment_path):
        experiment_count = len([f for f in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, f))])
        st.info(f"📊 실험 결과: {experiment_count}개 모델 실험 결과")
    else:
        st.warning("⚠️ 실험 결과 디렉토리가 없음")

    # 푸터
    st.markdown("---")
    st.markdown("*Attention Analysis App - Language Model Interpretability Research*")

elif page == "🤖 모델 로드":
    st.session_state.app_modules['model_loader'].show()

elif page == "📝 도메인 프롬프트 생성":
    st.session_state.app_modules['dataset_generator'].show()
elif page == "🔍 Evidence 추출":
    st.session_state.app_modules['evidence_extractor'].show()

elif page == "🔬 실험":
    st.session_state.app_modules['experiment'].show()

elif page == "📊 실험기록":
    st.session_state.app_modules['experiment_log'].show()

elif page == "📈 모니터링":
    st.session_state.app_modules['monitoring'].show()
