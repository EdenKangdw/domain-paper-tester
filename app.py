import streamlit as st
from tabs import model_loader, dataset_generator, monitoring
from utils import check_ollama_installed

# Ollama 설치 확인
is_ollama_ready, error_message = check_ollama_installed()
if not is_ollama_ready:
    st.error(error_message)
    st.stop()

# 사이드바 탭
tab = st.sidebar.radio("기능 선택", ["모델 로드", "데이터셋 생성", "모니터링"])

# 선택된 탭에 따라 해당 모듈의 show 함수 호출
if tab == "모델 로드":
    model_loader.show()
elif tab == "데이터셋 생성":
    dataset_generator.show()
elif tab == "모니터링":
    monitoring.show()
