import streamlit as st
from pathlib import Path
import json
from transformers import AutoTokenizer
from utils import check_ollama_model_status, OLLAMA_API_BASE
import requests
import random
from typing import List, Tuple
import re
from datetime import datetime
import time
import os

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치해주세요.")

# Model tokenizer settings
MODEL_TOKENIZER_MAP = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "gemma:2b": "google/gemma-2b",
    "gemma:7b": "google/gemma-7b",
    "qwen": "Qwen/Qwen-7B",
    "deepseek": "deepseek-ai/deepseek-coder-7b-base",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "yi": "01-ai/Yi-6B",
    "openchat": "openchat/openchat",
    "neural": "neural-chat/neural-chat-7b-v3-1",
    "phi": "microsoft/phi-2",
    "stable": "stabilityai/stable-code-3b"
}

def get_running_models():
    """Get list of currently running Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            running_models = []
            for model in models:
                if check_ollama_model_status(model["name"]):
                    running_models.append(model["name"])
            return running_models
        return []
    except:
        return []

def get_model_response(model_name, prompt):
    """Get response from the model"""
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        
        # 응답이 JSON인 경우
        try:
            json_response = response.json()
            if isinstance(json_response, dict) and "response" in json_response:
                return json_response["response"]
        except json.JSONDecodeError:
            pass
            
        # 응답이 바이트인 경우
        if isinstance(response.content, bytes):
            return response.content.decode('utf-8')
            
        return str(response.text)
    except Exception as e:
        return f"Error occurred: {str(e)}"

def tokenize_and_extract_words(text, tokenizer):
    """텍스트를 토큰화하고 단어를 추출"""
    print("\n=== Tokenization Debug ===")
    print(f"Original text: {text}")
    
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    
    # 구두점 제거를 위한 함수
    def clean_word(word):
        # 단어 끝의 구두점 제거 (쉼표, 마침표, 느낌표, 물음표 등)
        return word.rstrip(',.!?:;')
    
    # 단어 분리를 위한 함수
    def split_into_words(text):
        # 기본적인 단어 분리 (공백 기준)
        raw_words = []
        current_word = ""
        for char in text:
            if char.isspace():
                if current_word:
                    raw_words.append(current_word)
                    current_word = ""
            else:
                current_word += char
        if current_word:  # 마지막 단어 처리
            raw_words.append(current_word)
        
        # 구두점 제거 및 정리
        words = [clean_word(word) for word in raw_words]
        # 빈 문자열 제거
        words = [word for word in words if word.strip()]
        return words
    
    # 텍스트를 직접 단어로 분리
    words = split_into_words(text)
    print(f"Split words: {words}")
    
    # 각 단어의 시작과 끝 위치 찾기
    word_positions = []
    current_pos = 0
    text_lower = text.lower()
    
    for word in words:
        word_lower = word.lower()
        # 현재 위치부터 단어 찾기
        while current_pos < len(text):
            pos = text_lower.find(word_lower, current_pos)
            if pos != -1:
                word_positions.append((pos, pos + len(word), word))
                current_pos = pos + len(word)
                break
            current_pos += 1
    
    # 단어 위치를 기준으로 정렬
    word_positions.sort(key=lambda x: x[0])
    
    # 정렬된 순서로 단어 리스트 재구성
    words = [wp[2] for wp in word_positions]
    
    # 토큰과 단어 매핑
    word_to_tokens = {}
    current_token_pos = 0
    current_token_idx = 0
    
    for start, end, word in word_positions:
        token_indices = []
        token_text = ""
        
        while current_token_idx < len(tokens) and current_token_pos <= end:
            token = tokens[current_token_idx].replace('▁', '')
            if token.strip():
                token_text += token
                if current_token_pos >= start and current_token_pos < end:
                    token_indices.append(current_token_idx)
                current_token_pos += len(token)
            current_token_idx += 1
        
        if token_indices:
            word_to_tokens[word] = token_indices
    
    print("\n=== Word Processing Results ===")
    print(f"Final words: {words}")
    print("Word to token mapping:")
    for word, token_indices in word_to_tokens.items():
        token_values = [tokens[idx] for idx in token_indices]
        print(f"Word: {word}")
        print(f"Token indices: {token_indices}")
        print(f"Token values: {token_values}\n")
    
    return tokens, words, word_to_tokens

def format_word_and_token_info(tokens, words, word_to_tokens):
    """Format token and word information"""
    token_entries = [
        f'Token[{i}] = >>>{token}<<< (raw: {repr(token)})'
        for i, token in enumerate(tokens)
    ]
    
    word_entries = [
        f'Word[{i}] = >>>{word}<<<'
        for i, word in enumerate(words)
    ]
    
    return "\n".join(token_entries), "\n".join(word_entries)

def create_evidence_query(word_list, prompt, domain):
    """Evidence 추출을 위한 쿼리 생성"""
    # 단어 목록을 줄바꿈으로 분리하고 공백 제거
    words = [word.strip() for word in word_list.split('\n') if word.strip()]
    
    # 단어 목록을 표 형식으로 생성
    word_table = []
    for i, word in enumerate(words):
        word_table.append(f"| {i} | {word} |")
    word_table = "| Index | Word |\n|-------|------|\n" + "\n".join(word_table)

    return f"""You are a JSON API that extracts evidence tokens from text. Follow these instructions exactly:

1. From the token list below, identify tokens related to the '{domain}' domain.
2. Return ONLY a JSON object with this exact format:
{{
    "evidence_token_index": [numbers],
    "evidence": [tokens]
}}

Token List:
{word_table}

Input Text: "{prompt}"

Rules:
- evidence_token_index must be an array of numbers
- evidence must be an array of exact tokens from the list
- arrays must have the same length
- do not add any explanation or text outside the JSON
- do not modify or format the tokens
- do not use markdown or code blocks
"""

def extract_json_from_response(response):
    """Extract JSON from response"""
    import re
    # 문자열로 변환 보장
    if not isinstance(response, str):
        response = str(response)
    
    # 응답이 이미 JSON 객체인 경우
    if isinstance(response, dict):
        return response

    try:
        # 먼저 전체 응답을 JSON으로 파싱 시도
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            # JSON 형식의 문자열을 찾음 (앞뒤의 불필요한 텍스트 제거)
            json_match = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', response)
            if not json_match:
                raise ValueError("Could not find JSON format response")
            
            json_str = json_match.group(1)
            
            # JSON 문자열 정리
            # 1. 줄바꿈과 여러 공백을 단일 공백으로 변경
            json_str = re.sub(r'\s+', ' ', json_str).strip()
            
            # 2. 이스케이프되지 않은 큰따옴표를 찾아서 이스케이프 처리
            # 먼저 이미 이스케이프된 큰따옴표를 임시 치환
            json_str = json_str.replace('\\"', '___ESCAPED_QUOTE___')
            # 문자열 내의 이스케이프되지 않은 큰따옴표를 이스케이프 처리
            json_str = re.sub(r'(?<!\\)"([^"]*)"', r'"\1"', json_str)
            # 임시 치환된 이스케이프된 큰따옴표 복원
            json_str = json_str.replace('___ESCAPED_QUOTE___', '\\"')
            
            # 3. 작은따옴표로 감싸진 문자열을 큰따옴표로 변경
            json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
            
            # 4. 큰따옴표 주변의 공백 제거
            json_str = re.sub(r'\s*"\s*', '"', json_str)
            
            # 5. 콤마 주변의 공백 제거
            json_str = re.sub(r'\s*,\s*', ',', json_str)
            
            # 6. 중괄호 주변의 공백 제거
            json_str = re.sub(r'\s*{\s*', '{', json_str)
            json_str = re.sub(r'\s*}\s*', '}', json_str)
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse JSON: {response}")
            raise ValueError(f"JSON 파싱 실패: {str(e)}\n원본 응답: {response}")

def validate_evidence(result, words):
    """Validate evidence results"""
    required_fields = ["evidence_word_index", "evidence", "explanation"]
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise ValueError(f"Missing fields: {', '.join(missing_fields)}")
    
    evidence_word_index = result["evidence_word_index"]
    evidence = result["evidence"]
    
    if not isinstance(evidence_word_index, list):
        raise ValueError("evidence_word_index must be an array ([])")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be an array ([])")
    
    # 단어 목록 정리 (공백 제거)
    words = [word.strip() for word in words if word.strip()]
    
    # Validate indices
    invalid_indices = []
    for i, idx in enumerate(evidence_word_index):
        if not isinstance(idx, int):
            invalid_indices.append({"position": i, "index": idx, "reason": "not an integer"})
        elif not (0 <= idx < len(words)):
            invalid_indices.append({"position": i, "index": idx, "reason": f"out of range (0-{len(words)-1})"})
    
    if invalid_indices:
        details = [
            f"Position {e['position']}: Index {e['index']} ({e['reason']})"
            for e in invalid_indices
        ]
        raise ValueError(f"Invalid indices found:\n" + "\n".join(details))
    
    # Check if evidence and evidence_word_index lengths match
    if len(evidence) != len(evidence_word_index):
        raise ValueError(f"Array lengths don't match (evidence: {len(evidence)}, index: {len(evidence_word_index)})")
    
    # Check for words not in the list
    invalid_words = []
    for i, word in enumerate(evidence):
        if word not in words:
            invalid_words.append({
                "position": i,
                "word": word,
                "available_words": words
            })
    
    if invalid_words:
        details = [
            f"Position {w['position']}: '{w['word']}' (available words: {w['available_words']})"
            for w in invalid_words
        ]
        raise ValueError(f"Words not in list found:\n" + "\n".join(details))
    
    # Check index and word matching
    mismatches = []
    for i, (idx, word) in enumerate(zip(evidence_word_index, evidence)):
        if words[idx] != word:
            mismatches.append({
                "position": i,
                "index": idx,
                "expected": words[idx],
                "actual": word
            })
    
    if mismatches:
        details = [
            f"Position {m['position']}: Index {m['index']} should be '{m['expected']}' but got '{m['actual']}'"
            for m in mismatches
        ]
        raise ValueError(f"Index and word mismatches found:\n" + "\n".join(details))
    
    return evidence_word_index, evidence

def visualize_evidence(words, evidence_word_index, evidence, explanation):
    """Visualize evidence results"""
    highlighted_words = [
        f"<span style='background-color:#fff176; padding:2px'>{word}</span>"
        if i in evidence_word_index else word
        for i, word in enumerate(words)
    ]
    
    st.markdown("### Extracted Evidence:")
    st.markdown(" ".join(highlighted_words), unsafe_allow_html=True)
    
    # JSON 데이터를 먼저 파싱하고 검증
    try:
        json_data = {
            "evidence_word_index": evidence_word_index,
            "evidence": evidence,
            "explanation": explanation
        }
        # JSON 문자열로 변환했다가 다시 파싱하여 유효성 검사
        json_str = json.dumps(json_data, ensure_ascii=False)
        validated_data = json.loads(json_str)
        st.json(validated_data)
    except json.JSONDecodeError as e:
        st.error(f"JSON 데이터 오류: {str(e)}")
        st.code(str(json_data), language="json")

def get_test_prompt(domain: str) -> str:
    """도메인별 테스트 프롬프트를 반환합니다."""
    prompts = {
        "Medical": [
            "What are the main side effects of this medication?",
            "What are the contraindications for this treatment?",
            "What are the recommended dosages for this drug?",
            "What are the potential complications of this procedure?",
            "What are the warning signs to watch for?"
        ],
        "Legal": [
            "What are the key clauses in this contract?",
            "What are the main obligations of the parties?",
            "What are the termination conditions?",
            "What are the dispute resolution procedures?",
            "What are the confidentiality requirements?"
        ],
        "Technical": [
            "What is the main functionality of this code?",
            "What are the key features of this system?",
            "What are the system requirements?",
            "What are the performance specifications?",
            "What are the security measures implemented?"
        ],
        "General": [
            "What is the main content of this document?",
            "What are the key points discussed?",
            "What are the main conclusions?",
            "What are the important findings?",
            "What are the main recommendations?"
        ]
    }
    return random.choice(prompts.get(domain, ["Please enter your prompt here..."]))

def extract_evidence_with_ollama(prompt, tokens, model_name):
    """Ollama API를 사용하여 증거 추출"""
    try:
        # 토큰이 바이트 타입인 경우 문자열로 디코딩
        decoded_tokens = []
        for token in tokens:
            if isinstance(token, bytes):
                try:
                    decoded_tokens.append(token.decode('utf-8'))
                except UnicodeDecodeError:
                    try:
                        decoded_tokens.append(token.decode('latin-1'))
                    except:
                        decoded_tokens.append('')
            else:
                decoded_tokens.append(str(token))

        query = create_evidence_query("\n".join(decoded_tokens), prompt, "Medical")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": query,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                response_text = result['response']
                print(f"Raw response from model: {response_text}")  # 디버깅용
                
                try:
                    # JSON 문자열 정리
                    response_text = response_text.strip()
                    
                    # 이스케이프된 JSON 문자열 처리
                    if '\\"' in response_text:
                        # 이스케이프된 따옴표를 일반 따옴표로 변환
                        response_text = response_text.replace('\\"', '"')
                    
                    # JSON 객체 추출
                    json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                    else:
                        st.error("Could not find JSON object in response")
                        return [], []
                    
                    # JSON 파싱 시도
                    evidence_data = json.loads(response_text)
                    
                    # 필드명 정규화
                    evidence_data = {k.lower().replace('_', ''): v for k, v in evidence_data.items()}
                    
                    # 필수 필드 확인 (정규화된 필드명으로)
                    indices = evidence_data.get('evidencetokenindex', evidence_data.get('evidenceindices', []))
                    evidence = evidence_data.get('evidence', [])
                    
                    if not indices or not evidence:
                        st.error("Missing required fields in evidence data")
                        return [], []
                    
                    # 문장부호 제거 및 인덱스 조정
                    punctuation_pattern = re.compile(r'[^\w\s]')
                    filtered_indices = []
                    filtered_evidence = []
                    removed_count = 0
                    
                    for i, (idx, token) in enumerate(zip(indices, evidence)):
                        # 문장부호가 아닌 경우만 포함
                        if not punctuation_pattern.search(token):
                            # 이전에 제거된 토큰 수만큼 인덱스 조정
                            adjusted_idx = idx - removed_count
                            filtered_indices.append(adjusted_idx)
                            filtered_evidence.append(token)
                        else:
                            removed_count += 1
                    
                    indices = filtered_indices
                    evidence = filtered_evidence
                    
                    # 인덱스와 토큰 수가 일치하는지 확인
                    if len(indices) != len(evidence):
                        print(f"Debug - Indices length: {len(indices)}, Evidence length: {len(evidence)}")
                        print(f"Debug - Indices: {indices}")
                        print(f"Debug - Evidence: {evidence}")
                        st.error(f"Number of indices ({len(indices)}) and tokens ({len(evidence)}) do not match")
                        # 길이가 다를 경우 더 짧은 쪽에 맞춰 자르기
                        min_length = min(len(indices), len(evidence))
                        indices = indices[:min_length]
                        evidence = evidence[:min_length]
                    
                    # 인덱스가 유효한지 확인
                    if any(not isinstance(i, int) or i < 0 or i >= len(tokens) for i in indices):
                        st.error("Invalid indices found in response")
                        return [], []
                    
                    return indices, evidence
                except json.JSONDecodeError as e:
                    print(f"Error parsing response: {str(e)}\nResponse: {response_text}")
                    st.error(f"Evidence extraction failed: Invalid JSON format")
                    return [], []
            else:
                st.error("No response field in API result")
                return [], []
        else:
            st.error(f"API request failed with status code: {response.status_code}")
            return [], []
    except Exception as e:
        st.error(f"Error during evidence extraction: {str(e)}")
        return [], []

def load_tokenizer(model_key):
    """Load tokenizer for the given model"""
    try:
        # 모델 키에서 기본 모델명 추출 (예: gemma:7b -> gemma)
        base_model = model_key.split(":")[0]
        
        # MODEL_TOKENIZER_MAP에서 토크나이저 이름 찾기
        tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key)  # 전체 모델명으로 먼저 시도
        if not tokenizer_name:
            tokenizer_name = MODEL_TOKENIZER_MAP.get(base_model)  # 기본 모델명으로 시도
        
        if tokenizer_name:
            # Hugging Face 토큰 확인
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # 특별한 설정이 필요한 모델들
            if "qwen" in model_key.lower():
                if hf_token:
                    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=hf_token)
                else:
                    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            elif "gemma" in model_key.lower():
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                return tokenizer
            elif "llama" in model_key.lower():
                # Llama 모델은 토큰이 필요할 수 있음
                if hf_token:
                    return AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
                else:
                    st.warning("Llama 모델에 접근하려면 HUGGINGFACE_TOKEN 환경변수를 설정해주세요.")
                    return None
            else:
                if hf_token:
                    return AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
                else:
                    return AutoTokenizer.from_pretrained(tokenizer_name)
        
        st.error(f"Tokenizer not found for model {model_key}. Supported models: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
        return None
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

def generate_domain_prompt(domain: str, model_key: str) -> str:
    """도메인별로 새로운 프롬프트를 생성합니다."""
    prompt = f"""Generate a new question or prompt related to the {domain} domain.
The prompt should be:
1. Specific to the {domain} domain
2. Clear and concise
3. Focused on extracting key information
4. Natural and professional

Please provide only the prompt without any additional text or explanation."""

    try:
        response = get_model_response(model_key, prompt)
        # 응답에서 첫 번째 문장만 추출
        prompt = response.split('\n')[0].strip()
        return prompt
    except Exception as e:
        print(f"Error generating prompt: {str(e)}")
        return f"Please enter your {domain} domain prompt here..."

def show():
    st.title("Dataset Generator")
    
    # Model selection
    st.subheader("🤖 Model")
    
    # 모델 선택 방식
    model_selection_method = st.radio(
        "모델 선택 방식",
        ["Hugging Face 모델 (토크나이저 사용)", "Ollama 모델 (실행 중인 모델)"],
        horizontal=True,
        key="model_selection_method"
    )
    
    if model_selection_method == "Hugging Face 모델 (토크나이저 사용)":
        # Hugging Face 모델 목록
        hf_models = list(MODEL_TOKENIZER_MAP.keys())
        selected_model = st.selectbox(
            "Hugging Face 모델 선택",
            hf_models,
            key="hf_model_selector"
        )
        model_key = selected_model
        
        # 토크나이저 로드
        tokenizer = load_tokenizer(model_key)
        if not tokenizer:
            st.error(f"토크나이저를 로드할 수 없습니다: {model_key}")
            return
            
        st.success(f"✅ 토크나이저 로드 완료: {MODEL_TOKENIZER_MAP[model_key]}")
        
    else:
        # Ollama 모델 목록
        models = get_running_models()
        if not models:
            st.error("실행 중인 Ollama 모델이 없습니다. Model Load 탭에서 모델을 시작해주세요.")
            return
        
        selected_model = st.selectbox(
            "실행 중인 Ollama 모델 선택",
            models,
            key="ollama_model_selector"
        )
        model_key = selected_model.lower()
        
        # 토크나이저 로드
        tokenizer = load_tokenizer(model_key)
        if not tokenizer:
            st.warning(f"⚠️ 토크나이저를 찾을 수 없습니다: {model_key}")
            st.info("지원되는 모델: " + ", ".join(MODEL_TOKENIZER_MAP.keys()))
            return
    
    # Domain selection
    st.subheader("🎯 Domain")
    domains = ["Medical", "Legal", "Technical", "General"]
    
    # Dataset generation settings
    st.subheader("📊 Dataset Generation")
    generation_mode = st.radio(
        "Generation Mode",
        ["Single Domain", "All Domains"],
        help="단일 도메인 또는 모든 도메인에 대해 데이터셋을 생성합니다."
    )
    
    if generation_mode == "Single Domain":
        selected_domain = st.selectbox(
            "Select domain",
            domains,
            key="domain_selector"
        )
        selected_domains = [selected_domain]
    else:
        selected_domain = domains[0]  # 기본값으로 첫 번째 도메인 선택
        selected_domains = domains
    
    num_datasets = st.number_input(
        "Number of datasets per domain",
        min_value=1,
        max_value=100000,
        value=5,
        step=1,
        help="각 도메인별로 생성할 데이터셋의 개수"
    )
    
    # Generate dataset button
    if st.button("🔄 Generate Dataset", key="generate_dataset"):
        if not tokenizer:
            st.warning(f"⚠️ Tokenizer not found for model {model_key}. Supported models: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
            return
        
        # Ollama 모델인 경우 실행 상태 확인
        if model_selection_method == "Ollama 모델 (실행 중인 모델)":
            if not check_ollama_model_status(model_key):
                st.error(f"❌ Model {model_key} is not running. Please start it in the Model Load tab.")
                return
        
        try:
            total_domains = len(selected_domains)
            
            # 진행상황 표시를 위한 컨테이너들
            progress_text = st.empty()
            progress_counter = st.empty()
            time_info = st.empty()
            
            # 전체 시작 시간 기록
            total_start_time = time.time()
            
            for domain_idx, domain in enumerate(selected_domains, 1):
                # 도메인별 시작 시간 기록
                domain_start_time = time.time()
                
                progress_text.text(f"Generating datasets for {domain} domain...")
                
                with st.spinner(f"Generating {num_datasets} datasets for {domain} domain ({domain_idx}/{total_domains})..."):
                    # Create output directory
                    output_dir = Path(f"dataset/{model_key}/{domain.lower()}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create output file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = output_dir / f"{model_key}_{num_datasets}prompts_{timestamp}.jsonl"
                    
                    # Generate datasets
                    with open(output_path, "w", encoding="utf-8") as f:
                        for i in range(num_datasets):
                            # 진행상황 카운터 업데이트 (도메인 정보 포함)
                            progress_counter.text(f"{domain} domain: {i+1}/{num_datasets}")
                            
                            # 시간 정보 업데이트
                            current_time = time.time()
                            elapsed_time = current_time - total_start_time
                            
                            # 예상 시간 계산 (현재 진행률 기준)
                            total_datasets = total_domains * num_datasets
                            completed_datasets = (domain_idx - 1) * num_datasets + i
                            if completed_datasets > 0:
                                avg_time_per_dataset = elapsed_time / completed_datasets
                                remaining_datasets = total_datasets - completed_datasets
                                estimated_remaining_time = avg_time_per_dataset * remaining_datasets
                                estimated_total_time = elapsed_time + estimated_remaining_time
                                
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: {estimated_total_time:.1f}초 | 남은시간: {estimated_remaining_time:.1f}초")
                            else:
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: 계산 중...")
                            
                            # Generate new prompt for the domain
                            if model_selection_method == "Ollama 모델 (실행 중인 모델)":
                                prompt = generate_domain_prompt(domain, model_key)
                                response = get_model_response(model_key, prompt)
                            else:
                                # Hugging Face 모델의 경우 기본 프롬프트 사용
                                prompt = get_test_prompt(domain)
                                response = "Generated using Hugging Face tokenizer"
                            
                            # Tokenize and extract evidence
                            tokens = tokenizer.tokenize(prompt)
                            
                            if model_selection_method == "Ollama 모델 (실행 중인 모델)":
                                query = f"""Given the following text and prompt, extract evidence tokens that support the answer to the prompt.
Text: {prompt}
Prompt: {prompt}

Please provide the evidence in the following JSON format:
{{
    "evidence_token_index": [list of token indices],
    "evidence": [list of evidence tokens]
}}

Rules:
1. Only include tokens that directly support the answer
2. Maintain the original order of tokens
3. Include complete phrases or sentences
4. Do not include words unrelated to the domain"""

                                evidence_response = get_model_response(model_key, query)
                            else:
                                # Hugging Face 모델의 경우 기본 evidence 추출
                                evidence_indices, evidence_tokens = extract_evidence_with_ollama(prompt, tokens, "default")
                                evidence_response = json.dumps({
                                    "evidence_token_index": evidence_indices,
                                    "evidence": evidence_tokens
                                })
                            
                            try:
                                # Extract JSON part from response
                                if model_selection_method == "Ollama 모델 (실행 중인 모델)":
                                    json_match = re.search(r'(\{.*\})', evidence_response, re.DOTALL)
                                    if json_match:
                                        evidence_data = json.loads(json_match.group(1))
                                    else:
                                        evidence_data = {"evidence_token_index": [], "evidence": []}
                                else:
                                    evidence_data = json.loads(evidence_response)
                                
                                evidence_indices = evidence_data.get("evidence_token_index", [])
                                evidence_tokens = evidence_data.get("evidence", [])
                                
                                # Create output data
                                output = {
                                    "prompt": prompt,
                                    "response": response,
                                    "evidence_indices": evidence_indices,
                                    "evidence_tokens": evidence_tokens,
                                    "model": model_key,
                                    "domain": domain,
                                    "timestamp": timestamp,
                                    "index": i + 1
                                }
                                
                                # Write to file
                                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                                
                            except json.JSONDecodeError as e:
                                st.warning(f"Error parsing evidence response for dataset {i+1}: {str(e)}")
                                continue
                    
                    # 도메인별 완료 시간 계산
                    domain_elapsed_time = time.time() - domain_start_time
                    st.success(f"✅ Generated {num_datasets} datasets for {domain} domain (소요시간: {domain_elapsed_time:.1f}초)")
                    st.success(f"Dataset saved to {output_path}")
                
                # Add a small delay between domains
                if domain_idx < total_domains:
                    time.sleep(1)
            
            # 전체 완료 시간 계산
            total_elapsed_time = time.time() - total_start_time
            
            # 진행상황 컨테이너들 정리
            progress_text.empty()
            progress_counter.empty()
            time_info.empty()
            
            st.success(f"🎉 Completed generating datasets for all {total_domains} domains! (총 소요시간: {total_elapsed_time:.1f}초)")
                
        except Exception as e:
            st.error(f"Error during dataset generation: {str(e)}")
            return
    
    # Prompt input (for manual testing)
    st.subheader("✍️ Manual Testing")
    if model_selection_method == "Ollama 모델 (실행 중인 모델)":
        prompt = st.text_area(
            "Enter your prompt",
            value=generate_domain_prompt(selected_domain, model_key),
            height=150,
            key="prompt_input"
        )
    else:
        prompt = st.text_area(
            "Enter your prompt",
            value=get_test_prompt(selected_domain),
            height=150,
            key="prompt_input"
        )
    
    # Extract evidence button
    if st.button("🎯 Extract Evidence", key="extract_evidence"):
        if not tokenizer:
            st.warning(f"⚠️ Tokenizer not found for model {model_key}. Supported models: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
            return
        
        tokens = tokenizer.tokenize(prompt)
        with st.spinner("Extracting evidence..."):
            if model_selection_method == "Ollama 모델 (실행 중인 모델)":
                evidence_indices, evidence_tokens = extract_evidence_with_ollama(prompt, tokens, model_key)
            else:
                evidence_indices, evidence_tokens = extract_evidence_with_ollama(prompt, tokens, "default")
            
            if evidence_indices and evidence_tokens:
                st.markdown("### Extracted Evidence:")
                for idx, token in zip(evidence_indices, evidence_tokens):
                    st.markdown(f"- **{idx}**: {token}")