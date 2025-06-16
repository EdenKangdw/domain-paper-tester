import streamlit as st
from pathlib import Path
import json
from transformers import AutoTokenizer
from utils import check_ollama_model_status, OLLAMA_API_BASE
import requests
import random
from typing import List, Tuple
import re

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
        tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key.split(":")[0])
        if tokenizer_name:
            # Qwen 모델의 경우 trust_remote_code=True 옵션 추가
            if "qwen" in model_key.lower():
                return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_name)
        return None
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

def show():
    st.title("Dataset Generator")
    
    # Model selection
    st.subheader("🤖 Model")
    models = get_running_models()
    if not models:
        st.error("No running models found. Please start Ollama first.")
        return
    
    selected_model = st.selectbox(
        "Select a model",
        models,
        key="model_selector"
    )
    model_key = selected_model.lower()
    
    # Get tokenizer
    tokenizer = load_tokenizer(model_key)
    
    # Domain selection
    st.subheader("🎯 Domain")
    domain = st.selectbox(
        "Select domain",
        ["Medical", "Legal", "Technical", "General"],
        key="domain_selector"
    )
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt",
        value=get_test_prompt(domain),
        height=150,
        key="prompt_input"
    )
    
    # Preview section
    st.subheader("👀 Preview")
    if prompt.strip():
        # Extract evidence using Ollama
        if st.button("🎯 Extract Evidence", key="extract_evidence"):
            if not tokenizer:
                st.warning(f"⚠️ Tokenizer not found for model {model_key}. Supported models: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
                # 토크나이저 추가 버튼
                if st.button("➕ Add Tokenizer", help="현재 모델을 위한 토크나이저를 추가합니다"):
                    base_model = model_key.split(":")[0]
                    default_tokenizers = {
                        "mistral": "mistralai/Mistral-7B-v0.1",
                        "mixtral": "mistralai/Mixtral-8x7B-v0.1",
                        "llama2": "meta-llama/Llama-2-7b-hf",
                        "gemma": "google/gemma-7b",
                        "qwen": "Qwen/Qwen-7B",
                        "yi": "01-ai/Yi-6B",
                        "deepseek": "deepseek-ai/deepseek-coder-7b-base",
                        "openchat": "openchat/openchat",
                        "neural": "neural-chat/neural-chat-7b-v3-1",
                        "phi": "microsoft/phi-2",
                        "stable": "stabilityai/stable-code-3b"
                    }
                    if base_model in default_tokenizers:
                        if "MODEL_TOKENIZER_MAP" not in st.session_state:
                            st.session_state.MODEL_TOKENIZER_MAP = MODEL_TOKENIZER_MAP.copy()
                        st.session_state.MODEL_TOKENIZER_MAP[base_model] = default_tokenizers[base_model]
                        st.success(f"✅ Added tokenizer for {base_model}: {default_tokenizers[base_model]}")
                    else:
                        st.error(f"❌ No default tokenizer found for {base_model}")
            else:
                # Tokenize text
                tokens = tokenizer.tokenize(prompt)
                with st.spinner("Extracting evidence..."):
                    evidence_indices, evidence_tokens = extract_evidence_with_ollama(prompt, tokens, model_key)
                    if evidence_indices and evidence_tokens:
                        st.markdown("### Extracted Evidence:")
                        evidence_data = [
                            {"Index": idx, "Token": token, "Is Evidence": "✅"}
                            for idx, token in zip(evidence_indices, evidence_tokens)
                        ]
                        st.table(evidence_data)
                        
                        # 전체 토큰 목록에서 증거 토큰 하이라이트
                        st.markdown("### All Tokens:")
                        all_tokens_data = [
                            {"Index": i, "Token": token, "Is Evidence": "✅" if i in evidence_indices else ""}
                            for i, token in enumerate(tokens)
                        ]
                        st.table(all_tokens_data)
                    else:
                        st.warning("No evidence tokens found.")

    # Save section
    st.subheader("💾 Save")
    if st.button("📦 Save Evidence Extraction Results"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            # Double check if selected model is running
            if not check_ollama_model_status(model_key):
                st.error(f"❌ Model {model_key} is not running. Please start it in the Model Load tab.")
                st.stop()

            try:
                with st.spinner("Extracting and saving evidence..."):
                    # Get general response
                    response = get_model_response(model_key, prompt)

                    # Extract evidence
                    query = f"""Find words from the input prompt that are related to the '{domain}' domain.

Prompt: "{prompt}"

Word list:
{word_list}

Token information:
{token_list}

Important notes:
- Only find words from within the prompt
- Return empty arrays if no domain-related words are found
- Words must be used exactly as shown
- Do not modify or transform words
- Each word in the evidence array must exactly match a word from the word list

Response rules:
1. Only find words directly related to the '{domain}' domain from the prompt
2. Return empty arrays if no related words are found
3. evidence_word_index should only contain word numbers
4. evidence should contain exact copies of the words at those numbers
5. evidence_word_index and evidence arrays must have the same length

Response format:
{{
    "evidence_word_index": [word_number1, word_number2, ...],
    "evidence": ["word1", "word2", ...],
    "explanation": "Please explain why the selected words are related to the {domain} domain. If no related words are found, write 'No related words found.'"
}}

Validation:
1. Each number in evidence_word_index must be a valid word list index
2. Each word in evidence must match the word at its index
3. Words must be exact copies of the content between >>> and <<<
4. Do not include words unrelated to the domain"""

                    evidence_response = get_model_response(model_key, query)
                    try:
                        # Extract JSON part from response
                        import re
                        json_match = re.search(r'(\{[^{]*\})', evidence_response)
                        if not json_match:
                            raise ValueError("Could not find JSON format response")
                        
                        evidence_response = json_match.group(1)
                        result = json.loads(evidence_response)
                        
                        # Validate required fields
                        required_fields = ["evidence_word_index", "evidence", "explanation"]
                        missing_fields = [field for field in required_fields if field not in result]
                        if missing_fields:
                            raise ValueError(f"Missing fields: {', '.join(missing_fields)}")
                            
                        evidence_word_index = result["evidence_word_index"]
                        evidence = result["evidence"]
                        explanation = result.get("explanation", "")

                        # Validate list format
                        if not isinstance(evidence_word_index, list):
                            raise ValueError("evidence_word_index must be an array ([])")
                        if not isinstance(evidence, list):
                            raise ValueError("evidence must be an array ([])")

                        # Validate indices
                        invalid_indices = [i for i in evidence_word_index if not (isinstance(i, int) and 0 <= i < len(words))]
                        if invalid_indices:
                            raise ValueError(f"Invalid indices found: {invalid_indices}")

                        # Check if evidence and evidence_word_index lengths match
                        if len(evidence) != len(evidence_word_index):
                            raise ValueError(f"Array lengths don't match (evidence: {len(evidence)}, index: {len(evidence_word_index)})")

                        # Check if evidence matches actual words
                        mismatches = []
                        for i, idx in enumerate(evidence_word_index):
                            expected_word = words[idx]
                            actual_word = evidence[i]
                            if expected_word != actual_word:
                                mismatches.append({
                                    "position": i,
                                    "index": idx,
                                    "expected": repr(expected_word),
                                    "actual": repr(actual_word)
                                })
                        
                        if mismatches:
                            mismatch_details = [
                                f"Position {m['position']}: Index {m['index']} word mismatch (expected: {m['expected']}, got: {m['actual']})"
                                for m in mismatches
                            ]
                            raise ValueError(f"Word mismatches:\n" + "\n".join(mismatch_details))

                        # Save
                        output = {
                            "input": prompt,
                            "domain": domain,
                            "model_response": response,
                            "words": words,
                            "evidence_word_index": evidence_word_index,
                            "evidence": evidence,
                            "explanation": explanation
                        }

                        output_dir = Path("dataset_output")
                        output_dir.mkdir(exist_ok=True)
                        output_path = output_dir / f"{model_key}_{domain}.jsonl"
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(output, ensure_ascii=False) + "\n")

                        # Display results
                        st.success(f"🎉 Save complete: {output_path}")
                        
                        # Preview saved results
                        with st.expander("📋 View Saved Results"):
                            st.markdown("### Model Response:")
                            st.markdown(response)
                            
                            st.markdown("### Extracted Evidence:")
                            # Display results word by word
                            word_results = []
                            for i, word in enumerate(words):
                                is_evidence = i in evidence_word_index
                                word_results.append({
                                    "Index": i,
                                    "Word": word,
                                    "Is Evidence": "✅" if is_evidence else ""
                                })
                            st.table(word_results)
                            
                            st.markdown("### Evidence Explanation:")
                            st.markdown(explanation)
                            
                            st.markdown("### Complete Results:")
                            try:
                                json_data = {
                                    "evidence_word_index": evidence_word_index,
                                    "evidence": evidence,
                                    "explanation": explanation
                                }
                                json_str = json.dumps(json_data, ensure_ascii=False)
                                validated_data = json.loads(json_str)
                                st.json(validated_data)
                            except json.JSONDecodeError as e:
                                st.error(f"JSON 데이터 오류: {str(e)}")
                                st.code(str(json_data), language="json")

                    except json.JSONDecodeError as e:
                        st.error(f"Evidence extraction failed. JSON parsing error: {str(e)}")
                        st.code(evidence_response, language="text")
                    except ValueError as e:
                        st.error(f"Evidence extraction failed. Data validation error: {str(e)}")
                        st.code(evidence_response, language="text")
                    except Exception as e:
                        st.error(f"Error during evidence extraction: {str(e)}")
                        st.code(evidence_response, language="text")

            except Exception as e:
                st.error(f"❌ Ollama request failed: {e}")