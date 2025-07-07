import streamlit as st
from pathlib import Path
import json
from transformers import AutoTokenizer
from utils import check_ollama_model_status, get_available_models, OLLAMA_API_BASE, get_model_response
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

# Model tokenizer settings (파라미터 수 무시하고 기본 모델명만 사용)
MODEL_TOKENIZER_MAP = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "llama2:7b": "meta-llama/Llama-2-7b-hf",
    "gemma": "google/gemma-7b",
    "gemma:7b": "google/gemma-7b",
    "gemma:2b": "google/gemma-2b",
    "qwen": "Qwen/Qwen-7B",
    "qwen:7b": "Qwen/Qwen-7B",
    "qwen:latest": "Qwen/Qwen-7B",
    "deepseek": "deepseek-ai/deepseek-coder-7b-base",
    "deepseek:7b": "deepseek-ai/deepseek-coder-7b-base",
    "deepseek-r1": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-r1:7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-r1-distill-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral:7b": "mistralai/Mistral-7B-v0.1",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "yi": "01-ai/Yi-6B",
    "openchat": "openchat/openchat",
    "neural": "neural-chat/neural-chat-7b-v3-1",
    "phi": "microsoft/phi-2",
    "stable": "stabilityai/stable-code-3b"
}

# Origin 프롬프트 캐시
_origin_prompts_cache = {}

def clean_deepseek_response(response_text: str) -> str:
    """DeepSeek 모델의 <think> 태그와 그 안의 내용을 제거하고, 태그 뒤의 내용만 반환합니다."""
    if not response_text:
        return response_text
    
    print(f"Original DeepSeek response: {repr(response_text)}")  # 디버깅용
    
    # <think> 태그가 있는지 확인
    if "<think>" in response_text:
        # <think> 태그의 시작 위치 찾기
        think_start = response_text.find("<think>")
        
        # </think> 태그의 끝 위치 찾기
        think_end = response_text.find("</think>")
        
        print(f"Think tags found: start={think_start}, end={think_end}")  # 디버깅용
        
        if think_end != -1:
            # </think> 태그 이후의 내용만 추출 (</think>는 7자)
            after_think = response_text[think_end + 7:].strip()
            
            # 만약 </think> 이후에 내용이 있으면 그것을 사용
            if after_think:
                response_text = after_think
                print(f"Using after think content: {repr(after_think)}")  # 디버깅용
            else:
                # </think> 이후에 내용이 없으면 빈 문자열 반환 (재시도 유도)
                print("No content after think tags, returning empty")
                return ""
        else:
            # </think> 태그가 없으면 <think> 태그부터 끝까지 제거
            response_text = response_text[:think_start].strip()
            print(f"No closing tag, using before think: {repr(response_text)}")  # 디버깅용
    else:
        print(f"No think tags found, using original: {repr(response_text)}")  # 디버깅용
    
    # 최종 검증: 빈 문자열이면 빈 문자열 반환 (재시도 유도)
    if not response_text.strip():
        print("Cleaned response is empty, returning empty")
        return ""
    
    print(f"Final cleaned response: {repr(response_text)}")  # 디버깅용
    return response_text

def clean_prompt_text(prompt_text: str) -> str:
    """프롬프트 텍스트에서 불필요한 따옴표와 공백을 제거합니다."""
    if not prompt_text:
        return ""
    
    # 앞뒤 공백 제거
    prompt_text = prompt_text.strip()
    
    # 앞뒤 따옴표 제거 (중첩된 따옴표도 처리)
    while (prompt_text.startswith('"') and prompt_text.endswith('"')) or \
          (prompt_text.startswith("'") and prompt_text.endswith("'")):
        prompt_text = prompt_text[1:-1].strip()
    
    # 번호 제거 (예: "1. ", "2. ", "a) ", "b) " 등)
    import re
    prompt_text = re.sub(r'^\d+\.\s*', '', prompt_text)  # "1. ", "2. " 등 제거
    prompt_text = re.sub(r'^[a-z]\)\s*', '', prompt_text)  # "a) ", "b) " 등 제거
    prompt_text = re.sub(r'^[A-Z]\)\s*', '', prompt_text)  # "A) ", "B) " 등 제거
    prompt_text = re.sub(r'^[ivx]+\)\s*', '', prompt_text, flags=re.IGNORECASE)  # "i) ", "ii) " 등 제거
    
    # 불필요한 공백 정리
    prompt_text = re.sub(r'\s+', ' ', prompt_text)
    
    return prompt_text.strip()

def load_origin_prompts():
    """Origin 폴더에서 모델별, 도메인별 프롬프트를 로드합니다."""
    # 세션 상태에서 캐시 확인
    if "origin_prompts_cache" in st.session_state:
        return st.session_state.origin_prompts_cache
    
    origin_dir = Path("dataset/origin")
    if not origin_dir.exists():
        st.error("dataset/origin 폴더를 찾을 수 없습니다.")
        return {}
    
    prompts_cache = {}
    
    # 모델별 폴더 확인
    for model_dir in origin_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            prompts_cache[model_name] = {}
            
            # 도메인별 프롬프트 로드
            for domain in ["economy", "legal", "medical", "technical"]:
                domain_dir = model_dir / domain
                prompt_file = domain_dir / f"{domain}_prompts.json"
                
                if prompt_file.exists():
                    try:
                        with open(prompt_file, 'r', encoding='utf-8') as f:
                            prompts_data = json.load(f)
                            # 프롬프트 텍스트만 추출
                            prompts = [item["prompt"] for item in prompts_data]
                            prompts_cache[model_name][domain.capitalize()] = prompts
                    except Exception as e:
                        st.error(f"Error loading prompts for {model_name}/{domain}: {str(e)}")
                        prompts_cache[model_name][domain.capitalize()] = []
                else:
                    st.warning(f"Prompt file not found: {prompt_file}")
                    prompts_cache[model_name][domain.capitalize()] = []
    
    # 세션 상태에 캐시 저장
    st.session_state.origin_prompts_cache = prompts_cache
    return prompts_cache

@st.cache_data(ttl=60)  # 60초 캐시로 연장
def get_running_models():
    """Get list of currently running Ollama models with caching"""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            running_models = []
            
            # 병렬 처리를 위한 간단한 최적화
            for model in models:
                try:
                    # 더 빠른 상태 확인 (타임아웃 단축)
                    if check_ollama_model_status_fast(model["name"]):
                        running_models.append(model["name"])
                except:
                    continue  # 개별 모델 실패 시 건너뛰기
            
            return running_models
        return []
    except:
        return []

def check_ollama_model_status_fast(model_name):
    """빠른 모델 상태 확인 (타임아웃 단축)"""
    try:
        # 먼저 실행 중인 모델 목록 확인
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=3)
        if response.status_code == 200:
            running_models = response.json().get("models", [])
            for model in running_models:
                if model.get("name") == model_name:
                    return True
        
        # 실행 중인 모델 목록에 없으면 직접 테스트
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
    except Exception as e:
        print(f"Model status check error for {model_name}: {str(e)}")
        return False



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

def calculate_evidence_indices(evidence_tokens, all_tokens):
    """LLM이 추출한 evidence 토큰들의 인덱스를 계산합니다."""
    evidence_indices = []
    
    for evidence_token in evidence_tokens:
        # 토큰 목록에서 evidence 토큰의 인덱스 찾기
        found = False
        for i, token in enumerate(all_tokens):
            # 정확한 매칭 시도
            if token == evidence_token:
                evidence_indices.append(i)
                found = True
                break
            # 부분 매칭 시도 (토큰이 여러 부분으로 나뉜 경우)
            elif evidence_token in token or token in evidence_token:
                evidence_indices.append(i)
                found = True
                break
        
        # 매칭되지 않은 경우 로그 출력
        if not found:
            print(f"Warning: Could not find index for evidence token '{evidence_token}' in token list")
    
    return evidence_indices

def create_evidence_query(word_list, prompt, domain):
    """Evidence 추출을 위한 쿼리 생성 (기존 함수 유지 - 호환성)"""
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
    """도메인별 테스트 프롬프트를 origin 폴더에서 가져옵니다."""
    origin_prompts = load_origin_prompts()
    
    if domain in origin_prompts and origin_prompts[domain]:
        return random.choice(origin_prompts[domain])
    else:
        # fallback 프롬프트
        fallback_prompts = {
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
            "Economy": [
                "What is the main content of this document?",
                "What are the key points discussed?",
                "What are the main conclusions?",
                "What are the important findings?",
                "What are the main recommendations?"
            ]
        }
        return random.choice(fallback_prompts.get(domain, ["Please enter your prompt here..."]))

def extract_evidence_with_ollama(prompt, tokens, model_name, domain="economy"):
    """LLM이 evidence 토큰을 추출하고, 코드로 인덱스를 계산합니다."""
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

        # LLM에게 evidence 토큰 추출 요청
        query = create_evidence_query("\n".join(decoded_tokens), prompt, domain)
        
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
                
                # deepseek 모델의 <think> 태그 제거
                if "deepseek" in model_name.lower():
                    import re
                    # <think>...</think> 태그와 내용 제거
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                    # <think> 태그만 있는 경우 제거
                    response_text = re.sub(r'<think>\s*</think>', '', response_text)
                    response_text = response_text.strip()
                    print(f"Raw response from model (deepseek 태그 제거): {response_text}")  # 디버깅용
                else:
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
                        print("Could not find JSON object in response")
                        return [], []
                    
                    # JSON 파싱 시도
                    evidence_data = json.loads(response_text)
                    
                    # 필드명 정규화
                    evidence_data = {k.lower().replace('_', ''): v for k, v in evidence_data.items()}
                    
                    # LLM이 추출한 evidence 토큰 가져오기
                    evidence_tokens = evidence_data.get('evidence', [])
                    
                    if not evidence_tokens:
                        print("No evidence tokens found in LLM response")
                        return [], []
                    
                    # 코드로 evidence 토큰의 인덱스 계산
                    evidence_indices = calculate_evidence_indices(evidence_tokens, decoded_tokens)
                    
                    # 결과 검증
                    if not evidence_indices:
                        print(f"Warning: Could not find indices for evidence tokens: {evidence_tokens}")
                        return [], []
                    
                    # 인덱스가 유효한지 확인
                    if any(not isinstance(i, int) or i < 0 or i >= len(decoded_tokens) for i in evidence_indices):
                        print(f"Warning: Invalid indices found: {evidence_indices}")
                        return [], []
                    
                    print(f"LLM extracted {len(evidence_tokens)} evidence tokens: {evidence_tokens}")
                    print(f"Code calculated indices: {evidence_indices}")
                    return evidence_indices, evidence_tokens
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing response: {str(e)}\nResponse: {response_text}")
                    return [], []
            else:
                print("No response field in API result")
                return [], []
        else:
            print(f"API request failed with status code: {response.status_code}")
            return [], []
    except Exception as e:
        print(f"Error during evidence extraction: {str(e)}")
        return [], []

def load_tokenizer(model_key):
    """Load tokenizer for the given model with caching"""
    # 세션 상태에서 캐시된 토크나이저 확인
    cache_key = f"tokenizer_{model_key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        # 모델 키에서 기본 모델명 추출 (예: gemma:7b -> gemma, deepseek-r1:7b -> deepseek-r1)
        base_model = model_key.split(":")[0]
        
        # 디버깅 정보
        st.info(f"🔍 모델 키: {model_key}")
        st.info(f"🔍 기본 모델명: {base_model}")
        st.info(f"🔍 지원되는 모델들: {list(MODEL_TOKENIZER_MAP.keys())}")
        
        # MODEL_TOKENIZER_MAP에서 토크나이저 이름 찾기
        # 1. 전체 모델명으로 먼저 시도
        tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key)
        st.info(f"🔍 전체 모델명 매칭 시도: {model_key} -> {tokenizer_name}")
        
        if tokenizer_name:
            st.info(f"🔍 전체 모델명 매칭 성공: {model_key}")
        else:
            # 2. 기본 모델명으로 시도 (파라미터 수 무시)
            tokenizer_name = MODEL_TOKENIZER_MAP.get(base_model)
            st.info(f"🔍 기본 모델명 매칭 시도: {base_model} -> {tokenizer_name}")
            
            if tokenizer_name:
                st.info(f"🔍 기본 모델명 매칭 성공: {base_model}")
        
        st.info(f"🔍 최종 찾은 토크나이저: {tokenizer_name}")
        
        if tokenizer_name:
            # Hugging Face 토큰 확인
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # deepseek-r1 모델은 특별한 설정이 필요할 수 있음
            if "deepseek-r1" in model_key.lower():
                st.info(f"🔍 deepseek-r1 모델 토크나이저를 로드합니다.")
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            # 특별한 설정이 필요한 모델들
            elif "qwen" in model_key.lower():
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            elif "gemma" in model_key.lower():
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
            elif "llama" in model_key.lower():
                # Llama 모델은 토큰이 필요할 수 있음
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=hf_token)
                else:
                    st.warning("Llama 모델에 접근하려면 HUGGINGFACE_TOKEN 환경변수를 설정해주세요.")
                    return None
            else:
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            
            # 세션 상태에 토크나이저 캐시
            st.session_state[cache_key] = tokenizer
            return tokenizer
        
        st.error(f"Tokenizer not found for model {model_key}. Supported models: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
        return None
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

def generate_domain_prompt(domain: str, model_key: str) -> str:
    """도메인별로 origin 폴더에서 프롬프트를 가져옵니다."""
    origin_prompts = load_origin_prompts()
    
    # 모델별 프롬프트 확인
    if model_key in origin_prompts and domain in origin_prompts[model_key] and origin_prompts[model_key][domain]:
        return random.choice(origin_prompts[model_key][domain])
    else:
        # 도메인별 구체적인 요청 프롬프트
        domain_prompts = {
            "Medical": "Generate only one medical question a patient might ask a doctor. Respond with a single, clear question sentence only. Do not include explanations, lists, or any other text.",
            "Legal": "Generate only one legal question someone might ask a lawyer. Respond with a single, clear question sentence only. Do not include explanations, lists, or any other text.",
            "Technical": "Generate only one technical question about computers, software, or technology. Respond with a single, clear question sentence only. Do not include explanations, lists, or any other text.",
            "Economy": "Generate only one economic question about markets, finance, or business. Respond with a single, clear question sentence only. Do not include explanations, lists, or any other text."
        }
        
        request_prompt = domain_prompts.get(domain, f"Generate only one question about {domain}. Respond with a single, clear question sentence only. Do not include explanations, lists, or any other text.")
        
        try:
            response = get_model_response(model_key, request_prompt)
            
            # 응답이 빈 문자열이거나 비어있는 경우
            if not response:
                print(f"Empty or invalid response from model {model_key}")
                return ""
            
            # 응답 정리 및 검증
            if not response or not response.strip():
                print(f"Empty response from model {model_key}")
                return ""
            
            # 응답에서 첫 번째 문장만 추출
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            if not lines:
                print(f"No valid lines in response from model {model_key}")
                return ""
            
            prompt = lines[0]
            
            # 프롬프트 검증 (더 관대한 조건)
            if len(prompt) < 5:  # 최소 길이를 5자로 줄임
                print(f"Prompt too short: {prompt}")
                return ""
            
            # 명확히 잘못된 응답만 필터링
            invalid_starts = ('please enter', 'error', 'failed', 'i cannot', 'i am unable', 'i do not have')
            if prompt.lower().startswith(invalid_starts):
                print(f"Invalid prompt generated: {prompt}")
                return ""
            
            return prompt
        except Exception as e:
            print(f"Error generating prompt: {str(e)}")
            return ""

def show():
    st.title("📝 Domain Prompt Generator")
    st.markdown("도메인별 프롬프트를 생성하고 모델 응답을 받습니다.")
    
    # 강제 캐시 무효화 (개발 중에만 사용)
    if st.sidebar.button("🔄 강제 새로고침", key="force_refresh_dataset"):
        # 모든 캐시 무효화
        get_running_models.clear()
        get_available_models.clear()
        # 세션 상태 초기화
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith(('tokenizer_', 'origin_prompts_cache', 'generated_prompts', 'prompt_generation_complete'))]
        for key in keys_to_remove:
            del st.session_state[key]
        st.sidebar.success("강제 새로고침 완료!")
        st.rerun()
    
    # 세션 상태 초기화
    if 'dataset_generator_initialized' not in st.session_state:
        st.session_state.dataset_generator_initialized = True
    
    # ===== 모델 선택 섹션 =====
    st.markdown("---")
    st.subheader("🤖 Model Selection")
    
    # Ollama 모델 목록 (캐시됨)
    with st.spinner("🔄 사용 가능한 모델 목록을 확인하는 중..."):
        models = get_available_models()
    
    # 디버깅 정보 표시
    st.info(f"🔍 발견된 모델 수: {len(models)}개")
    if models:
        st.info(f"📋 모델 목록: {', '.join(models)}")
    
    if not models:
        st.error("❌ 사용 가능한 Ollama 모델이 없습니다. Model Load 탭에서 모델을 다운로드해주세요.")
        st.info("💡 해결 방법:")
        st.info("1. Model Load 탭에서 '🔄 새로고침' 버튼을 클릭하세요")
        st.info("2. 또는 이 페이지에서 '🔄 모델 목록 새로고침' 버튼을 클릭하세요")
        return
    
    # 모델 선택과 상태 표시를 컬럼으로 배치
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if models:
            st.markdown("**🔧 모델 선택 (여러 개 선택 가능)**")
            selected_models = st.multiselect(
                "사용 가능한 Ollama 모델들",
                models,
                default=[models[0]] if models else [],
                help="여러 모델을 선택하면 순차적으로 처리됩니다."
            )
            
            if selected_models:
                st.info(f"선택된 모델: {', '.join(selected_models)}")
                model_key = selected_models[0].lower()  # 첫 번째 모델을 기본값으로
            else:
                st.warning("⚠️ 최소 하나의 모델을 선택해주세요.")
                model_key = None
        else:
            st.warning("⚠️ 선택할 수 있는 모델이 없습니다.")
            model_key = None
            selected_models = []
    
    with col2:
        if selected_models:
            st.markdown("**📊 모델 상태 확인**")
            
            # 선택된 모델들의 상태 확인
            for model in selected_models:
                model_lower = model.lower()
                model_status = check_ollama_model_status_fast(model_lower)
                
                # 토크나이저 로드 상태 표시
                tokenizer = load_tokenizer(model_lower)
                
                if tokenizer:
                    if model_lower == model_key:  # 현재 선택된 모델
                        st.success(f"✅ {model} (토크나이저 로드됨)")
                    else:
                        st.info(f"ℹ️ {model} (토크나이저 로드됨)")
                else:
                    st.warning(f"⚠️ {model} (토크나이저 없음)")
                
                # 모델 실행 상태 표시
                if model_status:
                    st.success(f"🟢 {model} (실행 중)")
                else:
                    st.warning(f"🔴 {model} (미실행)")
        else:
            st.warning("⚠️ 모델을 먼저 선택해주세요.")
            tokenizer = None
    
    # ===== 도메인 설정 섹션 =====
    st.markdown("---")
    st.subheader("🎯 Domain Configuration")
    
    domains = ["Medical", "Legal", "Technical", "Economy"]
    
    # 생성 모드와 도메인 선택을 컬럼으로 배치
    col3, col4 = st.columns([1, 1])
    
    with col3:
        generation_mode = st.radio(
            "Generation Mode",
            ["Single Domain", "All Domains"],
            help="단일 도메인 또는 모든 도메인에 대해 데이터셋을 생성합니다."
        )
    
    with col4:
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
            st.info(f"모든 도메인 선택됨: {', '.join(domains)}")
    
    # ===== 섹션 1: 도메인 프롬프트 생성 =====
    st.markdown("---")
    st.markdown("## 🔥 STEP 1: Domain Prompt Generation")
    st.markdown("### 도메인별 프롬프트를 생성하고 모델 응답을 받습니다.")
    
    # 프롬프트 생성 설정
    col5, col6 = st.columns([1, 2])
    
    with col5:
        num_prompts = st.number_input(
            "Number of prompts per domain",
            min_value=1,
            max_value=10000,
            value=5,
            step=1,
            help="각 도메인별로 생성할 프롬프트의 개수 (최대 10000개)"
        )
    
    with col6:
        if generation_mode == "All Domains":
            total_prompts = len(domains) * num_prompts
            st.metric("총 생성될 프롬프트", f"{total_prompts}개")
        else:
            st.metric("생성될 프롬프트", f"{num_prompts}개")
    
    # 프롬프트 생성 버튼
    col7, col8 = st.columns([2, 1])
    
    with col7:
        generate_disabled = not model_key or not tokenizer
        generate_prompts_button = st.button("📝 Generate Prompts", type="primary", key="generate_prompts", disabled=generate_disabled)
    
    with col8:
        col8_1, col8_2 = st.columns(2)
        with col8_1:
            if st.button("🔄 모델 목록 새로고침", key="refresh_models_dataset"):
                get_available_models.clear()
                st.success("모델 목록이 새로고침되었습니다!")
        with col8_2:
            if st.button("🔍 모델 상태 확인", key="check_model_status"):
                if model_key:
                    # 캐시 무효화 후 상태 확인
                    get_running_models.clear()
                    status = check_ollama_model_status_fast(model_key)
                    if status:
                        st.success(f"✅ {model_key} 모델이 실행 중입니다!")
                    else:
                        st.warning(f"⚠️ {model_key} 모델이 실행되지 않고 있습니다.")
                        st.info("💡 Model Load 탭에서 모델을 시작한 후 다시 확인해주세요.")
                else:
                    st.warning("모델을 먼저 선택해주세요.")
    
    # 캐시 초기화 버튼
    if st.button("🗑️ 캐시 초기화", key="clear_cache_dataset"):
        # 캐시 함수들 초기화
        get_available_models.clear()
        # 세션 상태 캐시 초기화
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith(('tokenizer_', 'origin_prompts_cache'))]
        for key in keys_to_remove:
            del st.session_state[key]
        st.success("캐시가 초기화되었습니다!")
    
    # ===== 프롬프트 생성 실행 =====
    if generate_prompts_button:
        if not selected_models:
            st.error("❌ 최소 하나의 모델을 선택해주세요.")
            return
        
        # 선택된 모델들의 토크나이저와 상태 확인
        valid_models = []
        for model in selected_models:
            model_lower = model.lower()
            tokenizer = load_tokenizer(model_lower)
            model_status = check_ollama_model_status_fast(model_lower)
            
            if not tokenizer:
                st.warning(f"⚠️ Tokenizer not found for model {model}. Supported models: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
                continue
            
            if not model_status:
                st.warning(f"⚠️ Model {model}가 실행되지 않고 있습니다.")
                continue
            
            valid_models.append(model)
        
        if not valid_models:
            st.error("❌ 유효한 모델이 없습니다. 토크나이저와 실행 상태를 확인해주세요.")
            return
        
        try:
            total_models = len(valid_models)
            total_domains = len(selected_domains)
            
            # 진행상황 표시를 위한 컨테이너들
            progress_text = st.empty()
            progress_counter = st.empty()
            time_info = st.empty()
            
            # 전체 시작 시간 기록
            total_start_time = time.time()
            
            # 생성된 프롬프트들을 저장할 임시 데이터 (모델별로 구분)
            generated_prompts = {}
            
            # 모델별로 순차 처리
            for model_idx, model in enumerate(valid_models, 1):
                model_lower = model.lower()
                model_start_time = time.time()
                
                progress_text.text(f"Processing model {model} ({model_idx}/{total_models})...")
                
                # 모델별 프롬프트 저장소 초기화
                generated_prompts[model] = {}
                
                # 도메인별로 프롬프트 생성
                for domain_idx, domain in enumerate(selected_domains, 1):
                    domain_start_time = time.time()
                    
                    progress_text.text(f"Generating prompts for {model}/{domain} ({model_idx}/{total_models}, {domain_idx}/{total_domains})...")
                    
                    with st.spinner(f"Generating {num_prompts} prompts for {model}/{domain}..."):
                        # 도메인별 프롬프트 리스트 초기화
                        generated_prompts[model][domain] = []
                        used_prompts = set()  # 중복 제거를 위한 set
                        
                        i = 0
                        while len(generated_prompts[model][domain]) < num_prompts:
                            # 진행상황 카운터 업데이트
                            current_count = len(generated_prompts[model][domain])
                            progress_counter.text(f"{model}/{domain}: {current_count}/{num_prompts} (attempt {i+1})")
                            
                            # 시간 정보 업데이트
                            current_time = time.time()
                            elapsed_time = current_time - total_start_time
                            
                            # 예상 시간 계산
                            total_prompts_to_generate = total_models * total_domains * num_prompts
                            completed_prompts = (model_idx - 1) * total_domains * num_prompts + (domain_idx - 1) * num_prompts + i
                            if completed_prompts > 0:
                                avg_time_per_prompt = elapsed_time / completed_prompts
                                remaining_prompts = total_prompts_to_generate - completed_prompts
                                estimated_remaining_time = avg_time_per_prompt * remaining_prompts
                                estimated_total_time = elapsed_time + estimated_remaining_time
                                
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: {estimated_total_time:.1f}초 | 남은시간: {estimated_remaining_time:.1f}초")
                            else:
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: 계산 중...")
                            
                            # Generate new prompt for the domain
                            print(f"Generating prompt for {domain} with {model_lower}...")
                            prompt = generate_domain_prompt(domain, model_lower)
                            print(f"Generated prompt: {prompt}")
                            
                            # 프롬프트가 None이거나 유효하지 않은 경우 재시도
                            retry_count = 0
                            max_retries = 3
                            while (not prompt or prompt == "ERROR" or 
                                   prompt.lower().startswith(('please enter', 'error', 'failed', 'i cannot', 'i am unable'))) and retry_count < max_retries:
                                print(f"Invalid prompt generated, retrying... (attempt {retry_count + 1}/{max_retries})")
                                # 재시도 시에는 다른 요청 프롬프트 사용
                                if retry_count == 1:
                                    request_prompt = f"Ask a question about {domain.lower()} topics."
                                elif retry_count == 2:
                                    request_prompt = f"What would you like to know about {domain.lower()}?"
                                else:
                                    request_prompt = f"Generate a {domain.lower()} question."
                                
                                try:
                                    response = get_model_response(model_lower, request_prompt)
                                    if response and response.strip():
                                        lines = [line.strip() for line in response.split('\n') if line.strip()]
                                        if lines and len(lines[0]) >= 5:
                                            prompt = lines[0]
                                        else:
                                            prompt = ""
                                    else:
                                        prompt = ""
                                except:
                                    prompt = ""
                                retry_count += 1
                            
                            # 최대 재시도 후에도 유효하지 않은 경우 공백으로 처리
                            if not prompt or prompt == "ERROR" or prompt.lower().startswith(('please enter', 'error', 'failed', 'i cannot', 'i am unable')):
                                print(f"Failed to generate valid prompt after {max_retries} attempts, marking as empty")
                                prompt = ""
                            
                            # DeepSeek 모델의 경우 <think> 태그 제거 및 검증
                            if "deepseek" in model_lower:
                                original_prompt = prompt
                                prompt = clean_deepseek_response(prompt)
                                print(f"DeepSeek response cleaned: {prompt[:100]}...")  # 디버깅용
                                
                                # 공백이거나 <think> 태그만 남은 경우 다시 요청
                                if not prompt.strip() or prompt.strip().lower().startswith('<think>'):
                                    print(f"DeepSeek response is empty or only contains think tags, retrying...")
                                    prompt = ""
                            
                            # 모든 모델에 대해 프롬프트 텍스트 정리 (따옴표 제거)
                            prompt = clean_prompt_text(prompt)
                            print(f"Final prompt cleaned: {prompt[:100]}...")  # 디버깅용
                            
                            # 중복 제거: 이미 사용된 프롬프트인지 확인
                            if prompt in used_prompts:
                                print(f"Duplicate prompt detected: {prompt[:50]}...")
                                # 중복된 경우 다시 생성 시도
                                retry_count = 0
                                while prompt in used_prompts and retry_count < 3:
                                    print(f"Generating alternative prompt (attempt {retry_count + 1})")
                                    new_prompt = generate_domain_prompt(domain, model_lower)
                                    if new_prompt and new_prompt not in used_prompts:
                                        prompt = clean_prompt_text(new_prompt)
                                        break
                                    retry_count += 1
                            
                            # 공백인 경우 다시 요청 (무한 루프 방지를 위해 최대 10회)
                            retry_for_empty = 0
                            while (not prompt or not prompt.strip()) and retry_for_empty < 10:
                                print(f"Empty prompt generated, retrying... (attempt {retry_for_empty + 1}/10)")
                                prompt = generate_domain_prompt(domain, model_lower)
                                if prompt:
                                    # DeepSeek 모델의 경우 <think> 태그 제거 및 검증
                                    if "deepseek" in model_lower:
                                        original_prompt = prompt
                                        prompt = clean_deepseek_response(prompt)
                                        print(f"DeepSeek response cleaned: {prompt[:100]}...")  # 디버깅용
                                        
                                        # 공백이거나 <think> 태그만 남은 경우 다시 요청
                                        if not prompt.strip() or prompt.strip().lower().startswith('<think>'):
                                            print(f"DeepSeek response is empty or only contains think tags, retrying...")
                                            prompt = ""
                                            retry_for_empty += 1
                                            continue
                                    
                                    # 모든 모델에 대해 프롬프트 텍스트 정리 (따옴표 제거)
                                    prompt = clean_prompt_text(prompt)
                                    print(f"Final prompt cleaned: {prompt[:100]}...")  # 디버깅용
                                    
                                    # 중복 제거: 이미 사용된 프롬프트인지 확인
                                    if prompt in used_prompts:
                                        print(f"Duplicate prompt detected: {prompt[:50]}...")
                                        # 중복된 경우 다시 생성 시도
                                        retry_count = 0
                                        while prompt in used_prompts and retry_count < 3:
                                            print(f"Generating alternative prompt (attempt {retry_count + 1})")
                                            new_prompt = generate_domain_prompt(domain, model_lower)
                                            if new_prompt and new_prompt not in used_prompts:
                                                prompt = clean_prompt_text(new_prompt)
                                                break
                                            retry_count += 1
                                retry_for_empty += 1
                            
                            # 최종적으로 공백이 아닌 경우에만 저장
                            if prompt.strip():
                                # 프롬프트를 사용된 목록에 추가
                                used_prompts.add(prompt)
                                
                                # 프롬프트 정보 저장
                                prompt_data = {
                                    "prompt": prompt,
                                    "model": model_lower,
                                    "domain": domain,
                                    "index": len(generated_prompts[model][domain]) + 1
                                }
                                
                                generated_prompts[model][domain].append(prompt_data)
                                print(f"Successfully added prompt: {prompt[:50]}...")
                            else:
                                print(f"Failed to generate valid prompt after all retries, skipping...")
                            
                            i += 1
                        
                        # 도메인별 완료 시간 계산
                        domain_end_time = time.time()
                        domain_duration = domain_end_time - domain_start_time
                        print(f"{model}/{domain} prompts completed in {domain_duration:.2f} seconds")
                
                # 모델별 완료 시간 계산
                model_end_time = time.time()
                model_duration = model_end_time - model_start_time
                print(f"{model} model completed in {model_duration:.2f} seconds")
            
            # 전체 완료 시간 계산
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            
            # 진행상황 표시 정리
            progress_text.empty()
            progress_counter.empty()
            time_info.empty()
            
            # 생성된 프롬프트를 세션에 저장
            st.session_state.generated_prompts = generated_prompts
            st.session_state.prompt_generation_complete = True
            
            # ===== 프롬프트 파일 저장 =====
            st.markdown("---")
            st.subheader("💾 Saving Generated Prompts")
            
            # 모델별, 도메인별로 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            for model, model_prompts in generated_prompts.items():
                for domain, prompts in model_prompts.items():
                    if prompts:
                        # Create output directory (모델별 구조)
                        output_dir = Path(f"dataset/origin/{model.lower()}/{domain.lower()}")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create output file
                        output_path = output_dir / f"{domain.lower()}_{len(prompts)}prompts_{timestamp}.jsonl"
                        
                        # Save to file
                        with open(output_path, "w", encoding="utf-8") as f:
                            for prompt_data in prompts:
                                f.write(json.dumps(prompt_data, ensure_ascii=False) + "\n")
                        
                        saved_files.append((model, domain, output_path, len(prompts)))
            
            # ===== 저장 결과 표시 =====
            st.markdown("---")
            st.subheader("✅ Prompt Generation Complete!")
            
            # 결과 요약을 컬럼으로 배치
            col13, col14, col15 = st.columns(3)
            
            with col13:
                st.metric("총 소요 시간", f"{total_duration:.1f}초")
            
            with col14:
                st.metric("처리된 모델", f"{total_models}개")
            
            with col15:
                total_generated = total_models * total_domains * num_prompts
                st.metric("생성된 프롬프트", f"{total_generated}개")
            
            # 생성된 파일 목록 표시
            st.markdown("---")
            st.subheader("📋 Generated Files")
            
            for model, domain, file_path, count in saved_files:
                with st.container():
                    col16, col17 = st.columns([3, 1])
                    with col16:
                        st.write(f"📄 **{model}/{domain}**: `{file_path.name}` ({count}개 프롬프트)")
                    with col17:
                        st.write(f"📍 `{file_path.parent}`")
            
            # 출력 디렉토리 정보
            st.info(f"📁 모든 파일이 `dataset/origin/[모델명]/[도메인명]/` 디렉토리에 저장되었습니다.")
            st.success("🎉 프롬프트 생성이 완료되었습니다! 이제 Evidence 추출 페이지에서 evidence를 추출할 수 있습니다.")
            
        except Exception as e:
            st.markdown("---")
            st.subheader("❌ Prompt Generation Failed")
            
            # 에러 정보를 컬럼으로 배치
            col15, col16 = st.columns([2, 1])
            
            with col15:
                st.error(f"프롬프트 생성 중 오류가 발생했습니다:")
                st.code(str(e))
            
            with col16:
                st.warning("💡 해결 방법:")
                st.write("1. 모델이 실행 중인지 확인")
                st.write("2. 네트워크 연결 상태 확인")
                st.write("3. 캐시를 초기화하고 재시도")
            
            print(f"Prompt generation error: {str(e)}")
    
