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
import pandas as pd

def show():
    st.title("🧠 Ollama 모델 관리")
    
    # 세션 상태 초기화
    if 'installed_models' not in st.session_state:
        st.session_state.installed_models = get_available_models()
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    if 'show_model_list' not in st.session_state:
        st.session_state.show_model_list = False
    
    # 상단 설명
    st.markdown("""
    ### 📚 Ollama 모델 허브
    Ollama에서 제공하는 다양한 LLM 모델들을 확인하고 관리할 수 있습니다.
    """)
    
    # 설치된 모델 섹션
    st.markdown("### 🔄 설치된 모델")
    if st.button("새로고침", key="refresh_installed", type="primary"):
        st.session_state.installed_models = get_available_models()
    
    if st.session_state.installed_models:
        for model in st.session_state.installed_models:
            st.markdown(f"- `{model}`")
    else:
        st.info("설치된 모델이 없습니다.")
    
    # 모델 실행 섹션
    st.markdown("---")
    st.markdown("### 🚀 모델 실행")
    
    # 모델 선택
    col_select, col_info = st.columns([2, 1])
    
    with col_select:
        # 선택 방식 라디오 버튼
        select_method = st.radio(
            "모델 선택 방식",
            ["목록에서 선택", "직접 입력"],
            horizontal=True,
            key="select_method"
        )
        
        if select_method == "목록에서 선택":
            # 설치된 모델과 기본 모델을 구분하여 표시
            installed_models = st.session_state.installed_models
            default_models = ["llama2:7b", "llama2:13b", "llama2:70b", "gemma:2b", "gemma:7b", "qwen:7b", "qwen:14b"]
            
            # 설치된 모델과 기본 모델을 구분하여 옵션 생성
            model_options = []
            
            # 설치된 모델 추가
            if installed_models:
                model_options.append({"label": "--- 설치된 모델 ---", "disabled": True})
                for model in sorted(installed_models):
                    model_options.append({"label": f"📦 {model}", "value": model})
            
            # 기본 모델 추가
            model_options.append({"label": "--- 설치 가능한 모델 ---", "disabled": True})
            for model in sorted(default_models):
                if model not in installed_models:
                    model_options.append({"label": f"💫 {model}", "value": model})
            
            model_choice = st.selectbox(
                "실행할 모델을 선택하세요",
                options=[opt["value"] for opt in model_options if "value" in opt],
                format_func=lambda x: next((opt["label"] for opt in model_options if "value" in opt and opt["value"] == x), x),
                help="설치되지 않은 모델을 선택하면 자동으로 다운로드됩니다."
            )
        else:
            # 직접 입력 필드
            model_choice = st.text_input(
                "실행할 모델 코드를 입력하세요",
                placeholder="예: llama2:7b",
                help="모델명:버전 형식으로 입력하세요. (예: llama2:7b, gemma:2b)"
            )
    
    with col_info:
        if select_method == "목록에서 선택":
            st.markdown("""
            #### 아이콘 설명
            - 📦 : 이미 설치된 모델
            - 💫 : 설치 가능한 모델
            """)
        else:
            st.markdown("""
            #### 💡 입력 형식
            - 기본 형식: `모델명:버전`
            - 버전 표기: `7b`, `13b`, `70b` 등
            - 예시: `llama2:7b`, `gemma:2b`
            """)
    
    # 실행 버튼들을 나란히 배치
    col3, col4, col5 = st.columns(3)
    
    with col3:
        start_disabled = not model_choice  # 모델이 선택/입력되지 않은 경우 버튼 비활성화
        if st.button("🚀 모델 시작", type="primary", key="start_model", disabled=start_disabled):
            success, message = start_ollama_model(model_choice)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with col4:
        if st.button("🛑 모델 중지", type="secondary", key="stop_model"):
            success, message = stop_ollama_model()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with col5:
        if st.button("🔍 상태 확인", type="secondary", key="check_status"):
            is_running = check_ollama_model_status(model_choice)
            if is_running:
                st.success(f"✅ {model_choice} 모델이 실행 중입니다.")
            else:
                st.warning(f"❌ {model_choice} 모델이 실행되고 있지 않습니다.")
    
    # 사용 가능한 모델 목록 섹션 (별도 탭으로 분리)
    st.markdown("---")
    with st.expander("📚 사용 가능한 모델 목록", expanded=False):
        if st.button("모델 목록 가져오기", key="fetch_models", type="primary"):
            st.session_state.available_models = fetch_ollama_models()
            st.session_state.show_model_list = True
        
        # 모델 목록 표시
        if st.session_state.show_model_list and st.session_state.available_models:
            st.markdown("### 전체 모델 목록")
            
            # 모델 목록을 테이블 형태로 표시
            model_data = []
            for i, model_info in enumerate(st.session_state.available_models, 1):
                model_code = model_info['code']
                
                # 각 모델에 대한 행 추가
                model_data.append({
                    "번호": i,
                    "모델명": model_info["name"],
                    "파라미터": model_info['parameters'],
                    "설치": "✅" if model_info['code'] in st.session_state.installed_models else "❌",
                    "실행 코드": model_code
                })
            
            # 데이터프레임 생성 및 표시
            df = pd.DataFrame(model_data)
            
            # 테이블 스타일링을 위한 CSS
            st.markdown("""
            <style>
            .model-table {
                font-size: 14px;
                text-align: left;
            }
            .model-table th {
                background-color: #f0f2f6;
                padding: 8px;
            }
            .model-table td {
                padding: 8px;
            }
            .model-table tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # 테이블 표시
            for _, row in df.iterrows():
                col1, col2, col3, col4, col5 = st.columns([0.3, 1.2, 0.8, 0.4, 2.8])
                with col1:
                    st.write(f"#{row['번호']}")
                with col2:
                    st.write(f"`{row['모델명']}`")
                with col3:
                    st.write(row['파라미터'])
                with col4:
                    st.write(row['설치'])
                with col5:
                    st.code(row['실행 코드'], language="bash")
            
            # 사용 방법 설명
            st.info("""
            💡 '복사' 버튼을 클릭하면 모델 실행 코드가 클립보드에 복사됩니다.
            
            실행 예시:
            ```bash
            ollama run llama2:7b
            ```
            """)
    
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