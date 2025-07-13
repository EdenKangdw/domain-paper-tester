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
            timeout=15  # 타임아웃을 15초로 연장 (큰 모델 로딩 시간 고려)
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
        st.write("🛑 모델을 중지하는 중...")
        
        # 먼저 실행 중인 모델 목록 확인
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=5)
        if response.status_code != 200:
            return False, "Ollama 서버에 연결할 수 없습니다."
        
        running_models = response.json().get("models", [])
        if not running_models:
            return True, "실행 중인 모델이 없습니다."
        
        # 모든 실행 중인 모델 중지
        stopped_count = 0
        for model_info in running_models:
            model_name = model_info.get("name")
            if model_name:
                try:
                    # Ollama API를 사용해서 모델 중지 (더 안전한 방법)
                    # 빈 프롬프트로 짧은 요청을 보내서 모델을 종료시킴
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
                        stopped_count += 1
                        st.write(f"✅ {model_name} 모델 중지됨")
                    else:
                        st.write(f"⚠️ {model_name} 모델 중지 실패 (상태 코드: {stop_response.status_code})")
                        
                except Exception as e:
                    st.write(f"❌ {model_name} 모델 중지 중 오류: {str(e)}")
        
        if stopped_count > 0:
            return True, f"{stopped_count}개 모델이 성공적으로 중지되었습니다."
        else:
            return False, "모델 중지에 실패했습니다."
            
    except Exception as e:
        return False, f"예상치 못한 오류가 발생했습니다: {str(e)}"

def chat_with_model(model_name, prompt):
    """모델과 대화"""
    try:
        # requests 라이브러리를 사용하여 API 호출
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "응답을 받지 못했습니다.")
        else:
            return f"API 오류: {response.status_code}"
            
    except Exception as e:
        return f"오류 발생: {str(e)}"

@st.cache_data(ttl=30)  # 30초 캐시
def get_available_models():
    """Ollama에서 사용 가능한 모델 목록 조회 (캐시됨)"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
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
        'deepseek-r1-distill-llama': ['8B'],
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

def extract_evidence_with_ollama(prompt, tokens, model_key, domain):
    """
    Ollama 모델을 사용하여 프롬프트에서 evidence 토큰을 추출합니다.
    
    Args:
        prompt (str): 원본 프롬프트
        tokens (list): 토크나이저로 분리된 토큰 리스트
        model_key (str): 사용할 모델 이름
        domain (str): 도메인 이름
    
    Returns:
        tuple: (evidence_indices, evidence_tokens)
    """
    try:
        # 도메인별 evidence 추출 프롬프트 생성
        domain_prompts = {
            "Economy": "Find tokens related to economy, finance, market, investment, currency, and trade.",
            "Legal": "Find tokens related to law, regulations, contracts, rights, and obligations.",
            "Medical": "Find tokens related to medicine, health, disease, treatment, and drugs.",
            "Technical": "Find tokens related to technology, science, engineering, computers, and systems."
        }
        
        domain_instruction = domain_prompts.get(domain, "도메인 관련 중요한 토큰들을 찾아주세요.")
        
        # Evidence 추출을 위한 프롬프트 구성
        evidence_prompt = f"""
다음 프롬프트에서 {domain} 도메인과 관련된 evidence 토큰들을 추출해주세요.

{domain_instruction}

프롬프트: {prompt}

토큰 리스트: {tokens}

위 토큰 리스트에서 {domain} 도메인과 관련된 evidence 토큰들만 리스트로 응답해주세요.
예시: ["의학", "치료", "약물", "진단"]

응답 형식:
["토큰1", "토큰2", "토큰3", ...]
"""
        
        # Ollama API 호출
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_key,
                "prompt": evidence_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 100
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # 응답에서 토큰 리스트 추출
            import re
            import ast
            
            # JSON 형태의 리스트 추출 시도
            tokens_match = re.search(r'\[["\']([^"\']*(?:["\'][^"\']*["\'][^"\']*)*)["\']\]', response_text)
            
            if tokens_match:
                try:
                    # 전체 리스트를 파싱
                    list_match = re.search(r'\[[^\]]+\]', response_text)
                    if list_match:
                        evidence_tokens = ast.literal_eval(list_match.group())
                        if isinstance(evidence_tokens, list):
                            # 실제 토큰 리스트에서 인덱스 찾기
                            indices = []
                            for token in evidence_tokens:
                                if token in tokens:
                                    # 토큰의 모든 인덱스 찾기
                                    for i, t in enumerate(tokens):
                                        if t == token:
                                            indices.append(i)
                            
                            # 중복 제거 및 정렬
                            indices = sorted(list(set(indices)))
                            return indices, evidence_tokens
                except:
                    pass
            
            # 대안: 따옴표로 둘러싸인 토큰들 추출
            quoted_tokens = re.findall(r'["\']([^"\']+)["\']', response_text)
            if quoted_tokens:
                evidence_tokens = quoted_tokens
                # 실제 토큰 리스트에서 인덱스 찾기
                indices = []
                for token in evidence_tokens:
                    if token in tokens:
                        # 토큰의 모든 인덱스 찾기
                        for i, t in enumerate(tokens):
                            if t == token:
                                indices.append(i)
                
                # 중복 제거 및 정렬
                indices = sorted(list(set(indices)))
                return indices, evidence_tokens
            
            # 마지막 대안: 공백으로 구분된 단어들 추출
            words = re.findall(r'\b\w+\b', response_text)
            evidence_tokens = [word for word in words if word in tokens]
            if evidence_tokens:
                indices = []
                for token in evidence_tokens:
                    if token in tokens:
                        for i, t in enumerate(tokens):
                            if t == token:
                                indices.append(i)
                
                indices = sorted(list(set(indices)))
                return indices, evidence_tokens
            
            return [], []
        else:
            print(f"Ollama API 오류: {response.status_code}")
            return [], []
            
    except Exception as e:
        print(f"Evidence 추출 중 오류: {str(e)}")
        return [], []

def get_model_response(model_name, prompt):
    """모델에서 응답을 받아 프롬프트만 반환합니다 (response는 저장하지 않음)"""
    try:
        # 딥시크 모델은 더 짧은 응답과 빠른 타임아웃
        if "deepseek" in model_name.lower():
            num_predict = 80
            timeout = 10
        else:
            num_predict = 150
            timeout = 15
            
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": num_predict
                }
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # deepseek 모델의 <think> 태그 제거
            if "deepseek" in model_name.lower():
                import re
                # <think>...</think> 태그와 내용 제거
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                # <think> 태그만 있는 경우 제거
                response_text = re.sub(r'<think>\s*</think>', '', response_text)
                response_text = response_text.strip()
            
            # 응답이 비어있거나 의미없는 경우 빈 문자열 반환
            if not response_text or response_text.lower().startswith(('please enter', 'error', 'failed', 'i cannot', 'i am unable')):
                print(f"Invalid response from model {model_name}: {response_text}")
                return ""
            
            return response_text
        else:
            print(f"API 요청 실패: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"모델 응답 요청 중 오류: {str(e)}")
        return ""

@st.cache_data(ttl=30)  # 30초 캐시
def get_available_models():
    """Ollama에서 사용 가능한 모델 목록 조회 (캐시됨)"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
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
        'deepseek-r1': ['7B'],
        'solar': ['7B', '10.7B'],
        'meditron': ['7B'],
        'xwin': ['7B', '13B', '70B'],
        'tinyllama': ['1.1B'],
        'phi': ['2.7B'],
        'notus': ['7B'],
        'codellama': ['7B', '13B', '34B'],
        'wizardcoder': ['13B', '15B', '34B']
    }
    
    # 모델 이름에서 버전 정보 제거 (파라미터 수 무시)
    base_model = model_name.split(':')[0]
    
    # 기본 모델 이름으로 검색
    for key in model_params:
        if key.lower() == base_model.lower():
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

def extract_evidence_with_ollama(prompt, tokens, model_key, domain):
    """
    Ollama 모델을 사용하여 프롬프트에서 evidence 토큰을 추출합니다.
    
    Args:
        prompt (str): 원본 프롬프트
        tokens (list): 토크나이저로 분리된 토큰 리스트
        model_key (str): 사용할 모델 이름
        domain (str): 도메인 이름
    
    Returns:
        tuple: (evidence_indices, evidence_tokens)
    """
    try:
        # 도메인별 evidence 추출 프롬프트 생성
        domain_prompts = {
            "Economy": "Find tokens related to economy, finance, market, investment, currency, and trade.",
            "Legal": "Find tokens related to law, regulations, contracts, rights, and obligations.",
            "Medical": "Find tokens related to medicine, health, disease, treatment, and drugs.",
            "Technical": "Find tokens related to technology, science, engineering, computers, and systems."
        }
        
        domain_instruction = domain_prompts.get(domain, "도메인 관련 중요한 토큰들을 찾아주세요.")
        
        # Evidence 추출을 위한 프롬프트 구성
        evidence_prompt = f"""
다음 프롬프트에서 {domain} 도메인과 관련된 evidence 토큰들을 추출해주세요.

{domain_instruction}

프롬프트: {prompt}

토큰 리스트: {tokens}

위 토큰 리스트에서 {domain} 도메인과 관련된 evidence 토큰들만 리스트로 응답해주세요.
예시: ["의학", "치료", "약물", "진단"]

응답 형식:
["토큰1", "토큰2", "토큰3", ...]
"""
        
        # Ollama API 호출
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_key,
                "prompt": evidence_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 100
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # 응답에서 토큰 리스트 추출
            import re
            import ast
            
            # JSON 형태의 리스트 추출 시도
            tokens_match = re.search(r'\[["\']([^"\']*(?:["\'][^"\']*["\'][^"\']*)*)["\']\]', response_text)
            
            if tokens_match:
                try:
                    # 전체 리스트를 파싱
                    list_match = re.search(r'\[[^\]]+\]', response_text)
                    if list_match:
                        evidence_tokens = ast.literal_eval(list_match.group())
                        if isinstance(evidence_tokens, list):
                            # 실제 토큰 리스트에서 인덱스 찾기
                            indices = []
                            for token in evidence_tokens:
                                if token in tokens:
                                    # 토큰의 모든 인덱스 찾기
                                    for i, t in enumerate(tokens):
                                        if t == token:
                                            indices.append(i)
                            
                            # 중복 제거 및 정렬
                            indices = sorted(list(set(indices)))
                            return indices, evidence_tokens
                except:
                    pass
            
            # 대안: 따옴표로 둘러싸인 토큰들 추출
            quoted_tokens = re.findall(r'["\']([^"\']+)["\']', response_text)
            if quoted_tokens:
                evidence_tokens = quoted_tokens
                # 실제 토큰 리스트에서 인덱스 찾기
                indices = []
                for token in evidence_tokens:
                    if token in tokens:
                        # 토큰의 모든 인덱스 찾기
                        for i, t in enumerate(tokens):
                            if t == token:
                                indices.append(i)
                
                # 중복 제거 및 정렬
                indices = sorted(list(set(indices)))
                return indices, evidence_tokens
            
            # 마지막 대안: 공백으로 구분된 단어들 추출
            words = re.findall(r'\b\w+\b', response_text)
            evidence_tokens = [word for word in words if word in tokens]
            if evidence_tokens:
                indices = []
                for token in evidence_tokens:
                    if token in tokens:
                        for i, t in enumerate(tokens):
                            if t == token:
                                indices.append(i)
                
                indices = sorted(list(set(indices)))
                return indices, evidence_tokens
            
            return [], []
        else:
            print(f"Ollama API 오류: {response.status_code}")
            return [], []
            
    except Exception as e:
        print(f"Evidence 추출 중 오류: {str(e)}")
        return [], [] 