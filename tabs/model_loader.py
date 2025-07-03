import streamlit as st
from pathlib import Path
import json
import requests
from utils import (
    check_ollama_model_status,
    chat_with_model,
    get_available_models,
    fetch_ollama_models
)
import pandas as pd
import time
from datetime import datetime

# Ollama API 상수
OLLAMA_API_BASE = "http://localhost:11434"

def start_model_via_api(model_name: str) -> tuple[bool, str]:
    """Ollama API를 사용하여 모델을 시작합니다."""
    try:
        # 모델 실행 API 호출
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello",
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return True, f"Model {model_name} started successfully."
        else:
            return False, f"Failed to start model {model_name}. Status: {response.status_code}"
            
    except Exception as e:
        return False, f"Error starting model {model_name}: {str(e)}"

def stop_model_via_api(model_name: str) -> tuple[bool, str]:
    """Ollama API를 사용하여 모델을 중지합니다."""
    try:
        # 먼저 실행 중인 모델 목록 확인
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=5)
        
        if response.status_code != 200:
            return False, f"Failed to check model status. Status: {response.status_code}"
        
        running_models = response.json().get("models", [])
        running_model_names = [model["name"] for model in running_models]
        
        if model_name not in running_model_names:
            return True, f"Model {model_name} is not running."
        
        # 모델이 실행 중이면 중지 시도
        try:
            # Ollama API를 사용해서 모델 중지 (안전한 방법)
            stop_response = requests.post(
                f"{OLLAMA_API_BASE}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "stop",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": 0
                    }
                },
                timeout=3
            )
            
            if stop_response.status_code == 200:
                return True, f"Model {model_name} stopped successfully."
            else:
                return False, f"Failed to stop model {model_name}. Status: {stop_response.status_code}"
                
        except Exception as e:
            return False, f"Error stopping model {model_name}: {str(e)}"
            
    except Exception as e:
        return False, f"Error checking model status: {str(e)}"

def get_running_models_via_api() -> list[str]:
    """Ollama API를 사용하여 실행 중인 모델 목록을 가져옵니다."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception as e:
        print(f"Error fetching running models: {e}")
        return []

def show():
    # 페이지 제목
    st.header("🤖 Model Loader")
    st.markdown("Ollama 모델을 관리하고 실행할 수 있습니다.")
    
    # 세션 상태 초기화
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = 0
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = ""
    if 'test_response' not in st.session_state:
        st.session_state.test_response = ""
    if 'test_prompt' not in st.session_state:
        st.session_state.test_prompt = ""
    
    # ===== 1. 설치된 모델 섹션 =====
    st.markdown("### 📦 Installed Models")
    st.markdown("Currently installed models in the system.")
    
    # 설치된 모델 목록 (캐시됨)
    installed_models = get_available_models()
    
    # 새로고침 버튼
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh", type="primary", key="refresh_installed"):
            get_available_models.clear()
            st.session_state.last_refresh_time = time.time()
            st.success("Model list refreshed!")
    
    # 마지막 새로고침 시간 표시
    if st.session_state.last_refresh_time > 0:
        with col2:
            st.caption(f"Last refresh: {datetime.fromtimestamp(st.session_state.last_refresh_time).strftime('%H:%M:%S')}")
    
    # 설치된 모델 목록 표시
    if installed_models:
        for model in installed_models:
            st.markdown(f"- `{model}`")
    else:
        st.info("No models installed.")
    
    st.markdown("---")
    
    # ===== 2. 실행 중인 모델 섹션 =====
    st.markdown("### 🚀 Running Models")
    st.markdown("Currently running models.")
    
    # 실행 중인 모델 목록 조회
    running_models = get_running_models_via_api()
    
    col3, col4 = st.columns([1, 4])
    with col3:
        if st.button("🔄 Check Running", type="secondary", key="refresh_running"):
            st.rerun()
    
    with col4:
        st.caption(f"Found {len(running_models)} running model(s)")
    
    # 실행 중인 모델 목록 표시
    if running_models:
        for model in running_models:
            st.markdown(f"- 🟢 `{model}` (Running)")
    else:
        st.info("No models are currently running.")
    
    st.markdown("---")
    
    # ===== 3. 모델 실행 섹션 =====
    st.markdown("### 🎯 Model Execution")
    st.markdown("Select and run a model.")
    
    # 모델 선택
    col_select, col_info = st.columns([2, 1])
    
    with col_select:
        # 선택 방식 라디오 버튼
        select_method = st.radio(
            "Selection Method",
            ["From List", "Direct Input"],
            horizontal=True,
            key="select_method"
        )
        
        if select_method == "From List":
            # 설치된 모델과 기본 모델을 구분하여 표시
            default_models = ["llama2:7b", "llama2:13b", "llama2:70b", "gemma:2b", "gemma:7b", "qwen:7b", "qwen:14b"]
            
            # 설치된 모델과 기본 모델을 구분하여 옵션 생성
            model_options = []
            
            # 설치된 모델 추가
            if installed_models:
                model_options.append({"label": "--- Installed Models ---", "disabled": True})
                for model in sorted(installed_models):
                    status = "🟢 (Running)" if model in running_models else "⚪ (Stopped)"
                    model_options.append({"label": f"📦 {model} {status}", "value": model})
            
            # 기본 모델 추가
            model_options.append({"label": "--- Available Models ---", "disabled": True})
            for model in sorted(default_models):
                if model not in installed_models:
                    model_options.append({"label": f"💫 {model}", "value": model})
            
            model_choice = st.selectbox(
                "Select a model to run",
                options=[opt["value"] for opt in model_options if "value" in opt],
                format_func=lambda x: next((opt["label"] for opt in model_options if "value" in opt and opt["value"] == x), x),
                help="Select a model to run. Uninstalled models will be downloaded automatically.",
                key="model_selectbox"
            )
        else:
            # 직접 입력 필드
            model_choice = st.text_input(
                "Enter model name",
                placeholder="e.g., llama2:7b",
                help="Enter model name in format: model:version (e.g., llama2:7b, gemma:2b)",
                key="model_text_input"
            )
        
        # 선택된 모델을 세션에 저장
        if model_choice:
            st.session_state.model_choice = model_choice
    
    with col_info:
        if select_method == "From List":
            st.markdown("""
            #### Icon Legend
            - 📦 : Installed models
            - 💫 : Available models
            - 🟢 : Running
            - ⚪ : Stopped
            """)
        else:
            st.markdown("""
            #### 💡 Input Format
            - Format: `model:version`
            - Versions: `7b`, `13b`, `70b`, etc.
            - Examples: `llama2:7b`, `gemma:2b`
            """)
    
    # 실행/중지 버튼들
    col5, col6, col7 = st.columns(3)
    
    with col5:
        start_disabled = not st.session_state.model_choice
        if st.button("🚀 Start Model", type="primary", key="start_model", disabled=start_disabled):
            if st.session_state.model_choice in running_models:
                st.warning(f"Model {st.session_state.model_choice} is already running.")
            else:
                with st.spinner(f"Starting {st.session_state.model_choice}..."):
                    success, message = start_model_via_api(st.session_state.model_choice)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    with col6:
        if st.button("🛑 Stop Model", type="secondary", key="stop_model"):
            if st.session_state.model_choice:
                with st.spinner(f"Stopping {st.session_state.model_choice}..."):
                    success, message = stop_model_via_api(st.session_state.model_choice)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.warning(message)
            else:
                st.warning("Please select a model first.")
    
    with col7:
        if st.button("🔍 Check Status", type="secondary", key="check_status"):
            if st.session_state.model_choice:
                is_running = check_ollama_model_status(st.session_state.model_choice)
                if is_running:
                    st.success(f"✅ {st.session_state.model_choice} is running.")
                else:
                    st.warning(f"❌ {st.session_state.model_choice} is not running.")
            else:
                st.warning("Please select a model first.")
    
    st.markdown("---")
    
    # ===== 4. 모델 테스트 섹션 =====
    st.markdown("### 🧪 Model Testing")
    st.markdown("Test the selected model's performance.")
    
    if st.session_state.model_choice:
        st.info(f"Selected model: **{st.session_state.model_choice}**")
        
        test_prompt = st.text_area(
            "Enter test prompt", 
            height=100,
            placeholder="e.g., 'Hello! Let's have a simple conversation.'",
            key="test_prompt_input"
        )
        
        col_test1, col_test2 = st.columns([1, 4])
        with col_test1:
            if st.button("💫 Run Test", type="primary", key="test_button"):
                if not test_prompt:
                    st.warning("Please enter a prompt.")
                else:
                    # 먼저 모델 상태 확인
                    if not check_ollama_model_status(st.session_state.model_choice):
                        st.error(f"❌ {st.session_state.model_choice} is not running. Please start the model first.")
                    else:
                        with st.spinner("Generating model response..."):
                            response = chat_with_model(st.session_state.model_choice, test_prompt)
                            st.session_state.test_response = response
        with col_test2:
            st.caption("💡 The model must be running to test.")
        
        # 응답이 있으면 표시
        if st.session_state.test_response:
            st.markdown("---")
            st.markdown("### 🤖 Model Response:")
            st.markdown(st.session_state.test_response)
    else:
        st.warning("Please select a model first.")
    
    st.markdown("---")
    
    # ===== 5. 사용 가능한 모델 목록 섹션 =====
    st.markdown("### 📚 Available Models")
    st.markdown("All models available in Ollama.")
    
    if st.button("📋 Show All Models", type="secondary", key="show_all_models"):
        with st.spinner("Fetching available models..."):
            models_list = fetch_ollama_models()
            if models_list is not None and len(models_list) > 0:
                # 리스트를 DataFrame으로 변환
                import pandas as pd
                models_df = pd.DataFrame(models_list)
                st.dataframe(models_df, use_container_width=True)
            else:
                st.warning("Failed to fetch models or no models available.")
    
    # ===== 6. 시스템 관리 섹션 =====
    st.markdown("---")
    st.markdown("### ⚙️ System Management")
    st.markdown("Manage system cache and settings.")
    
    # 캐시 초기화 버튼
    if st.button("🗑️ Clear Cache", help="Clear cache while keeping model selection", type="secondary", key="clear_cache"):
        get_available_models.clear()
        keys_to_remove = ['last_refresh_time']
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache cleared! (Model selection is preserved)") 