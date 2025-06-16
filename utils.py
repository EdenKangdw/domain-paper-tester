import streamlit as st
import requests
import json
import subprocess
import re
from subprocess import PIPE, Popen
import threading
import queue
import time
import shutil

OLLAMA_API_BASE = "http://localhost:11434"
LOG_FILE = "ollama_output.log"

def check_ollama_installed():
    """Ollama 설치 및 실행 상태 확인"""
    # which 명령어로 ollama 실행 파일 확인
    ollama_path = shutil.which('ollama')
    if not ollama_path:
        return False, """
### ⚠️ Ollama가 설치되어 있지 않습니다!

WSL에 Ollama를 설치하려면 다음 명령어를 실행하세요:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

설치 후 Ollama 서비스를 시작하세요:
```bash
ollama serve
```
        """
    
    # Ollama 서비스가 실행 중인지 확인
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=2)
        return True, None
    except:
        return False, """
### ⚠️ Ollama 서비스가 실행되고 있지 않습니다!

다음 명령어로 Ollama 서비스를 시작하세요:
```bash
ollama serve
```
        """

def clean_ansi(text):
    """ANSI 이스케이프 시퀀스 및 특수 문자 제거"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = re.sub(r'^.*\r(?!$)', '', text)
    text = ansi_escape.sub('', text)
    text = re.sub(r'[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]', '', text)
    text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_progress(text):
    """다운로드 진행률 포맷팅"""
    if "pulling" in text.lower():
        match = re.search(r'pulling ([^:]+): (\d+)%', text.lower())
        if match:
            component, percentage = match.groups()
            bar_length = 20
            filled_length = int(bar_length * int(percentage) / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            return f"📥 {component}:\n{bar} {percentage}%"
    return text

def stream_process_output(process, queue):
    """프로세스의 출력을 실시간으로 큐에 전달"""
    for line in iter(process.stdout.readline, ''):
        queue.put(line)
    process.stdout.close()

def show_terminal_output(process, timeout=60):
    """터미널 출력을 실시간으로 표시하는 함수"""
    st.markdown('<style>div[data-testid="stCodeBlock"] > div { width: 100% !important; }</style>', unsafe_allow_html=True)
    log_container = st.empty()
    progress_container = st.empty()
    log_text = []
    last_update = ""
    
    output_queue = queue.Queue()
    output_thread = threading.Thread(
        target=stream_process_output, 
        args=(process, output_queue),
        daemon=True
    )
    output_thread.start()
    
    start_time = time.time()
    while time.time() - start_time < timeout and output_thread.is_alive():
        try:
            while True:
                try:
                    line = output_queue.get_nowait()
                    if line:
                        clean_line = clean_ansi(line)
                        if clean_line and clean_line != last_update:
                            formatted_line = format_progress(clean_line)
                            log_text.append(formatted_line)
                            last_update = clean_line
                            if len(log_text) > 20:
                                log_text.pop(0)
                            log_container.code('\n'.join(log_text))
                except queue.Empty:
                    break
            
            time.sleep(0.1)
            progress_container.text(f"진행 중... ({int(time.time() - start_time)}초)")
            
        except Exception as e:
            st.error(f"출력 처리 중 오류 발생: {str(e)}")
            break
    
    progress_container.empty()
    return log_text

def check_ollama_model_status(model_name):
    """모델 상태 확인"""
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": "test",
                "stream": False
            },
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def start_ollama_model(model_name):
    """모델 시작"""
    try:
        st.write("🚀 모델을 실행하는 중...")
        
        process = Popen(
            f"ollama run {model_name} 2>&1 | tee {LOG_FILE}",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            bufsize=1,
            universal_newlines=True
        )
        
        log_text = show_terminal_output(process)
        
        if check_ollama_model_status(model_name):
            return True, "모델이 성공적으로 시작되었습니다."
        else:
            return False, "모델 실행은 시작되었으나, 아직 준비되지 않았습니다. 잠시 후 다시 확인해주세요."
                
    except Exception as e:
        return False, f"예상치 못한 오류가 발생했습니다: {str(e)}"

def stop_ollama_model():
    """모델 중지"""
    try:
        st.write("🛑 모델 프로세스를 종료하는 중...")
        
        # 실행 중인 ollama 프로세스 찾기
        process = Popen(
            "ps aux | grep 'ollama run' | grep -v grep | awk '{print $2}'",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True
        )
        pid = process.stdout.read().strip()
        
        if pid:
            # 프로세스 종료
            kill_process = Popen(
                f"kill {pid}",
                shell=True,
                stdout=PIPE,
                stderr=PIPE,
                text=True
            )
            kill_process.wait()
            
            if kill_process.returncode == 0:
                return True, "모델이 성공적으로 중지되었습니다."
            else:
                return False, "모델 프로세스 종료 중 오류가 발생했습니다."
        else:
            return True, "실행 중인 모델이 없습니다."
            
    except Exception as e:
        return False, f"예상치 못한 오류가 발생했습니다: {str(e)}"

def chat_with_model(model_name, prompt):
    """모델과 대화"""
    try:
        response_container = st.empty()
        current_response = []

        process = Popen(
            f"curl -s -N -X POST http://localhost:11434/api/generate -d '{{\"model\":\"{model_name}\",\"prompt\":\"{prompt}\"}}'",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        current_response.append(data['response'])
                        response_container.markdown(''.join(current_response))
                except json.JSONDecodeError:
                    continue
        
        process.stdout.close()
        process.wait()
        
        if current_response:
            return ''.join(current_response)
        else:
            return '응답을 받지 못했습니다.'
            
    except Exception as e:
        st.error(f"요청 중 오류 발생: {str(e)}")
        return f"오류 발생: {str(e)}"

def get_available_models():
    """Ollama에서 사용 가능한 모델 목록 조회"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

def get_model_parameters(model_name):
    """모델의 파라미터 수를 반환"""
    model_params = {
        'llama2': ['7B', '13B', '70B'],
        'llama2-uncensored': ['7B', '13B'],
        'mistral': ['7B'],
        'mixtral': ['8x7B'],
        'gemma': ['2B', '7B'],
        'qwen': ['7B', '14B', '72B'],
        'yi': ['6B', '34B'],
        'openchat': ['7B'],
        'neural': ['7B'],
        'falcon': ['7B', '40B'],
        'dolphin': ['7B'],
        'vicuna': ['7B', '13B'],
        'zephyr': ['7B'],
        'nous-hermes': ['7B', '13B'],
        'orca': ['3B', '13B'],
        'starling': ['7B'],
        'openhermes': ['7B', '13B'],
        'wizard': ['7B', '13B'],
        'stable-beluga': ['7B', '13B'],
        'samantha': ['7B'],
        'phind': ['34B'],
        'deepseek': ['7B', '67B'],
        'solar': ['7B', '10.7B'],
        'meditron': ['7B'],
        'xwin': ['7B', '13B', '70B'],
        'tinyllama': ['1.1B'],
        'phi': ['2.7B'],
        'notus': ['7B'],
        'codellama': ['7B', '13B', '34B'],
        'wizardcoder': ['13B', '15B', '34B']
    }
    
    # 모델 이름에서 버전 정보 제거
    base_model = model_name.split(':')[0]
    
    # 기본 모델 이름으로 검색
    for key in model_params:
        if key in base_model.lower():
            return model_params[key]
    
    return []

def fetch_ollama_models():
    """Ollama 허브에서 사용 가능한 LLM 모델 목록 가져오기"""
    try:
        st.write("🔍 Ollama LLM 모델 목록을 가져오는 중...")
        
        response = requests.get("https://ollama.com/library", timeout=10)
        response.raise_for_status()
        
        # HTML 응답에서 모델 이름 추출
        models = re.findall(r'"/library/([^"]+)"', response.text)
        
        # LLM이 아닌 모델들 필터링
        excluded_keywords = [
            'coder', 'code', 'instruct', 'solar', 'phi', 
            'neural-chat', 'wizard-math', 'dolphin', 
            'stablelm', 'starcoder', 'wizardcoder'
        ]
        
        # LLM 모델만 필터링하고 파라미터 정보 추가
        llm_models = []
        for model in models:
            # 제외할 키워드가 모델 이름에 포함되어 있는지 확인
            if not any(keyword in model.lower() for keyword in excluded_keywords):
                params = get_model_parameters(model)
                # 파라미터 정보가 있는 모델만 추가
                if params:
                    base_name = model.split(':')[0]
                    # 각 파라미터 버전별로 별도의 항목 추가
                    for param in params:
                        param_code = param.lower().replace('x', '')  # 8x7B -> 7b
                        model_code = f"{base_name}:{param_code}"
                        llm_models.append({
                            'name': base_name,
                            'code': model_code,
                            'parameters': param
                        })
        
        return sorted(llm_models, key=lambda x: (x['name'], x['parameters']))
        
    except Exception as e:
        st.error(f"모델 목록 가져오기 실패: {str(e)}")
        return [] 