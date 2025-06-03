import streamlit as st
from pathlib import Path
import json
import requests
from utils import (
    check_ollama_model_status,
    start_ollama_model,
    stop_ollama_model,
    chat_with_model,
    get_available_models,
    fetch_ollama_models
)

def show():
    st.title("🧠 Ollama 모델 실행 도우미")
    
    # 설치된 모델 목록 가져오기
    installed_models = get_available_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 설치된 모델 목록 새로고침"):
            installed_models = get_available_models()
            if installed_models:
                st.success(f"설치된 모델: {', '.join(installed_models)}")
            else:
                st.info("설치된 모델이 없습니다.")
    
    with col2:
        if st.button("📚 Ollama 허브 모델 목록 보기"):
            available_models = fetch_ollama_models()
            if available_models:
                st.success(f"사용 가능한 모델 수: {len(available_models)}개")
                # 모델 목록을 여러 열로 표시
                cols = st.columns(3)
                for i, model in enumerate(available_models):
                    cols[i % 3].markdown(f"- `{model}`")
            else:
                st.warning("모델 목록을 가져올 수 없습니다.")
    
    st.markdown("---")
    
    # 모델 선택 (설치된 모델 + 기본 모델)
    default_models = ["llama2", "gemma", "qwen", "deepseek"]
    all_models = sorted(set(installed_models + default_models))
    model_choice = st.selectbox(
        "실행할 모델을 선택하세요",
        all_models,
        help="설치되지 않은 모델을 선택하면 자동으로 다운로드됩니다."
    )
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        if st.button("🚀 모델 시작"):
            success, message = start_ollama_model(model_choice)
            if not success:
                st.error(message)
    
    with col4:
        if st.button("🛑 모델 중지"):
            success, message = stop_ollama_model()
            if not success:
                st.error(message)
    
    with col5:
        if st.button("🔍 모델 상태 확인"):
            is_running = check_ollama_model_status(model_choice)
            if is_running:
                st.success(f"✅ {model_choice} 모델이 실행 중입니다.")
            else:
                st.warning(f"❌ {model_choice} 모델이 실행되고 있지 않습니다.")
    
    st.markdown("---")
    
    # 모델 테스트
    st.subheader("🧪 모델 테스트")
    st.info(f"현재 선택된 모델: **{model_choice}**")
    
    test_prompt = st.text_area("테스트할 프롬프트를 입력하세요", height=100)
    
    if st.button("💫 테스트 실행"):
        if not test_prompt:
            st.warning("프롬프트를 입력해주세요.")
        else:
            # 먼저 모델 상태 확인
            if not check_ollama_model_status(model_choice):
                st.error(f"❌ {model_choice} 모델이 실행되고 있지 않습니다. 먼저 모델을 시작해주세요.")
            else:
                response = chat_with_model(model_choice, test_prompt)
                st.markdown("### 🤖 모델 응답:")
                st.markdown(response) 