import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
import requests
from transformers import AutoTokenizer
import re
import ast
from typing import List, Tuple, Optional, Dict, Any

# 상수 정의
OLLAMA_API_BASE = "http://localhost:11434"
TIMEOUT = 30

# 모델별 토크나이저 매핑
MODEL_TOKENIZER_MAP = {
    "gemma:7b": "google/gemma-7b",
    "mistral:7b": "mistralai/Mistral-7B-v0.1",
    "qwen:7b": "Qwen/Qwen-7B",
    "llama2:7b": "meta-llama/Llama-2-7b-hf",
    "llama2:13b": "meta-llama/Llama-2-13b-hf",
    "llama2:70b": "meta-llama/Llama-2-70b-hf",
    "deepseek-r1:7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-r1-distill-llama:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}

# 도메인별 evidence 추출 프롬프트
DOMAIN_PROMPTS = {
    "economy": "Find tokens related to economy, finance, market, investment, currency, and trade.",
    "legal": "Find tokens related to law, regulations, contracts, rights, and obligations.",
    "medical": "Find tokens related to medicine, health, disease, treatment, and drugs.",
    "technical": "Find tokens related to technology, science, engineering, computers, and systems."
}

@st.cache_data(ttl=300)
def get_available_models() -> List[str]:
    """사용 가능한 Ollama 모델 목록을 가져옵니다."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

def check_model_status(model_key: str) -> bool:
    """특정 모델의 실행 상태를 확인합니다."""
    try:
        # 먼저 실행 중인 모델 목록 확인
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=5)
        if response.status_code == 200:
            running_models = response.json().get("models", [])
            model_names = [model["name"] for model in running_models]
            if model_key in model_names:
                return True
        
        # 실행 중이 아니라면 모델이 사용 가능한지 테스트
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_key,
                "prompt": "test",
                "stream": False,
                "options": {
                    "num_predict": 1
                }
            },
            timeout=10
        )
        
        if response.status_code == 200:
            # 응답이 성공하면 모델이 사용 가능함
            return True
        elif response.status_code == 404:
            # 모델이 존재하지 않음
            return False
        else:
            # 기타 오류
            return False
            
    except Exception as e:
        print(f"모델 상태 확인 중 오류: {str(e)}")
        return False

def get_running_models() -> List[str]:
    """실행 중인 모델 목록을 가져옵니다."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE}/api/ps", timeout=5)
        if response.status_code == 200:
            running_models = response.json().get("models", [])
            return [model["name"] for model in running_models]
        return []
    except Exception as e:
        print(f"실행 중인 모델 목록 조회 중 오류: {str(e)}")
        return []

def load_file_data(file_path: Path) -> List[Dict[str, Any]]:
    """파일에서 데이터를 로드합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix == ".json":
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            elif file_path.suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return []

def load_origin_prompts(domain: str) -> List[Dict[str, Any]]:
    """도메인별 origin 프롬프트 파일들을 로드합니다."""
    origin_dir = Path(f"dataset/origin/{domain.lower()}")
    if not origin_dir.exists():
        return []
    
    all_prompts = []
    
    # 모든 JSON 및 JSONL 파일 처리
    for file_path in origin_dir.glob("*.json*"):
        file_data = load_file_data(file_path)
        all_prompts.extend(file_data)
    
    return all_prompts

def find_token_positions(prompt: str, tokens: List[str], tokenizer) -> List[int]:
    """프롬프트에서 토큰들의 위치를 찾습니다 (토큰 단위 인덱스)."""
    try:
        # 프롬프트를 토큰화
        tokenized = tokenizer.tokenize(prompt)
        token_indices = []
        
        # 각 evidence 토큰에 대해 토큰 인덱스 찾기
        for evidence_token in tokens:
            evidence_token_lower = evidence_token.lower()
            found_index = -1  # 기본값 -1 (찾지 못함)
            
            # 토큰화된 리스트에서 매칭되는 토큰 찾기
            for i, token in enumerate(tokenized):
                # Ġ 문자 제거하고 토큰을 디코딩하여 원본 텍스트와 비교
                clean_token = token.replace('Ġ', '')
                decoded_token = tokenizer.convert_tokens_to_string([token]).strip()
                
                # 디버깅을 위한 로그 추가
                print(f"   매칭 시도: evidence='{evidence_token_lower}' vs clean_token='{clean_token.lower()}' vs decoded='{decoded_token.lower()}'")
                
                # 세 가지 방법 모두 시도
                if (clean_token.lower() == evidence_token_lower or 
                    decoded_token.lower() == evidence_token_lower or
                    token.lower() == evidence_token_lower):
                    found_index = i
                    print(f"   ✅ 매칭 성공: {evidence_token_lower} -> 인덱스 {i}")
                    break  # 첫 번째 매칭만 찾기
            
            token_indices.append(found_index)
        
        return token_indices
        
    except Exception as e:
        print(f"토큰 위치 찾기 중 오류: {str(e)}")
        return [-1] * len(tokens)  # 모든 토큰에 대해 -1 반환

def extract_tokens_from_response(response_text: str) -> Optional[List[str]]:
    """모델 응답에서 토큰 리스트를 추출합니다."""
    try:
        print(f"   🔍 토큰 추출 시작: {len(response_text)} 문자 응답")
        
        # JSON 형태의 리스트 추출 시도
        import re
        import ast
        
        # JSON 배열 패턴 찾기
        list_match = re.search(r'\[[^\]]+\]', response_text)
        if list_match:
            print(f"   JSON 배열 패턴 발견: {list_match.group()}")
            try:
                evidence_tokens = ast.literal_eval(list_match.group())
                if isinstance(evidence_tokens, list):
                    # 문자열만 필터링
                    result = [str(token).strip() for token in evidence_tokens if token]
                    print(f"   ✅ JSON 배열에서 추출: {len(result)}개 토큰")
                    return result
            except Exception as json_error:
                print(f"   ❌ JSON 파싱 실패: {str(json_error)}")
                pass
        
        # 따옴표로 둘러싸인 토큰들 추출
        print(f"   따옴표 패턴으로 추출 시도...")
        quoted_tokens = re.findall(r'["\']([^"\']+)["\']', response_text)
        if quoted_tokens:
            result = [token.strip() for token in quoted_tokens if token.strip()]
            print(f"   ✅ 따옴표 패턴에서 추출: {len(result)}개 토큰")
            return result
        
        # 공백으로 구분된 단어들 추출 (마지막 수단)
        print(f"   단어 패턴으로 추출 시도...")
        words = re.findall(r'\b\w+\b', response_text)
        if words:
            result = [word.strip() for word in words if word.strip()]
            print(f"   ✅ 단어 패턴에서 추출: {len(result)}개 토큰")
            return result
        
        print(f"   ❌ 모든 추출 방법 실패")
        return None
        
    except Exception as e:
        print(f"❌ 토큰 추출 중 오류: {str(e)}")
        print(f"   응답 텍스트: {response_text[:100]}...")
        return None

def call_ollama_api(model_key: str, prompt: str) -> Optional[str]:
    """Ollama API를 호출합니다."""
    try:
        print(f"   📡 Ollama API 호출: {model_key}")
        print(f"   프롬프트 길이: {len(prompt)} 문자")
        
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_key,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # 더 낮은 temperature로 일관된 응답
                    "top_p": 0.1,        # 더 낮은 top_p로 확실한 응답
                    "num_predict": 50,    # 짧은 응답으로 제한
                    "repeat_penalty": 1.1, # 반복 방지
                    "top_k": 1           # 가장 확실한 토큰만 선택
                }
            },
            timeout=TIMEOUT
        )
        
        print(f"   API 응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # deepseek 모델의 <think> 태그 제거
            if "deepseek" in model_key.lower():
                import re
                # <think>...</think> 태그와 내용 제거
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                # <think> 태그만 있는 경우 제거
                response_text = re.sub(r'<think>\s*</think>', '', response_text)
                response_text = response_text.strip()
                print(f"   ✅ API 호출 성공 (deepseek 태그 제거): {len(response_text)} 문자 응답")
            else:
                print(f"   ✅ API 호출 성공: {len(response_text)} 문자 응답")
            
            return response_text
        else:
            print(f"❌ Ollama API 오류: {response.status_code}")
            print(f"   응답 내용: {response.text[:200]}...")
            return None
    except Exception as e:
        print(f"❌ API 호출 중 오류: {str(e)}")
        print(f"   모델: {model_key}")
        print(f"   프롬프트 길이: {len(prompt)}")
        return None

def extract_evidence_tokens(prompt: str, model_key: str, domain: str) -> Tuple[List[int], List[str]]:
    """
    모델을 사용하여 프롬프트에서 evidence 토큰을 추출합니다.
    
    Args:
        prompt (str): 원본 프롬프트
        model_key (str): 사용할 모델 이름
        domain (str): 도메인 이름
    
    Returns:
        tuple: (evidence_indices, evidence_tokens)
    """
    try:
        print(f"🔍 Evidence 추출 시작: {domain} 도메인")
        print(f"   프롬프트 길이: {len(prompt)} 문자")
        
        domain_instruction = DOMAIN_PROMPTS.get(domain.lower(), "Find important tokens related to the domain.")
        print(f"   도메인 지시사항: {domain_instruction}")
        
        # Evidence 추출을 위한 프롬프트 구성 (강력한 영어 응답 강제)
        evidence_prompt = f"""
You are an English-only evidence extraction system. Your task is to extract English tokens from the given prompt.

DOMAIN: {domain}
INSTRUCTION: {domain_instruction}

INPUT PROMPT: "{prompt}"

CRITICAL RULES:
1. You MUST respond ONLY in English
2. You MUST extract ONLY English words that exist in the input prompt
3. You MUST return ONLY a JSON array format
4. You MUST NOT translate words
5. You MUST NOT add words that are not in the prompt
6. You MUST NOT respond in Korean or any other language
7. You MUST focus on domain-specific medical/technical terms

EXAMPLES:
- For medical domain: ["clinical", "findings", "diagnosis", "viral", "encephalitis", "adults"]
- For legal domain: ["legal", "contract", "liability", "jurisdiction"]
- For technical domain: ["algorithm", "implementation", "optimization"]

RESPONSE FORMAT (JSON array only):
["word1", "word2", "word3"]
"""
        
        print(f"   Ollama API 호출 시작...")
        # Ollama API 호출
        response_text = call_ollama_api(model_key, evidence_prompt)
        if not response_text:
            print(f"❌ No response from Ollama API for model {model_key}")
            return [], []
        
        print(f"   Ollama API 응답 받음: {len(response_text)} 문자")
        print(f"   응답 미리보기: {response_text[:200]}...")
        
        # 응답에서 토큰 리스트 추출
        print(f"   토큰 추출 시작...")
        evidence_tokens = extract_tokens_from_response(response_text)
        if not evidence_tokens:
            print(f"❌ Failed to extract tokens from response: {response_text[:100]}...")
            return [], []
        
        print(f"🔍 Extracted {len(evidence_tokens)} tokens from response: {evidence_tokens}")
        
        # 영어 토큰만 필터링
        print(f"   영어 토큰 필터링 시작...")
        english_tokens = []
        for token in evidence_tokens:
            token_clean = token.lower().strip()
            # 영어 단어만 허용 (한국어, 특수문자 제외)
            if token_clean and all(c.isascii() and c.isalnum() or c.isspace() for c in token_clean):
                english_tokens.append(token_clean)
        
        print(f"   영어 토큰 필터링 결과: {len(english_tokens)}개")
        
        if not english_tokens:
            print(f"❌ No valid English tokens found after filtering")
            return [], []
        
        print(f"🔍 Found {len(english_tokens)} valid English tokens: {english_tokens}")
        
        # 프롬프트를 소문자로 변환하여 매칭
        print(f"   프롬프트 매칭 시작...")
        prompt_lower = prompt.lower()
        valid_tokens = []
        
        for token in english_tokens:
            # 프롬프트에 실제로 존재하는지 확인 (단어 경계 고려)
            if token in prompt_lower:
                # 원본 프롬프트에서 정확한 토큰 찾기
                import re
                pattern = r'\b' + re.escape(token) + r'\b'
                if re.search(pattern, prompt_lower):
                    valid_tokens.append(token)
        
        print(f"   프롬프트 매칭 결과: {len(valid_tokens)}개")
        
        if not valid_tokens:
            print(f"❌ No tokens matched in prompt after word boundary check")
            return [], []
        
        print(f"🔍 Found {len(valid_tokens)} tokens that match in prompt: {valid_tokens}")
        
        # 토크나이저 로드 (선택적)
        print(f"   토크나이저 로드 시작...")
        tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key)
        print(f"{model_key} : {tokenizer_name}")
        
        if tokenizer_name:
            try:
                tokenizer = load_tokenizer_cached(tokenizer_name)
                print(f"   토크나이저 로드 완료: {tokenizer_name}")
                
                # 토큰 위치 찾기 (토큰 단위 인덱스)
                print(f"   토큰 위치 찾기 시작...")
                indices = find_token_positions(prompt, valid_tokens, tokenizer)
                print(f"   토큰 위치 찾기 결과: {indices}")
                
                # 유효한 인덱스만 필터링 (-1이 아닌 것들)
                valid_indices = [idx for idx in indices if idx != -1]
                valid_tokens_with_indices = [token for i, token in enumerate(valid_tokens) if indices[i] != -1]
                
                print(f"🔍 Final result: {len(valid_indices)} indices, {len(valid_tokens_with_indices)} tokens")
                print(f"   최종 evidence 토큰: {valid_tokens_with_indices}")
                print(f"   최종 evidence 인덱스: {valid_indices}")
                
                return valid_indices, valid_tokens_with_indices
                
            except Exception as e:
                print(f"   ⚠️ 토크나이저 로드 실패, 더미 인덱스 사용: {str(e)}")
                # 토크나이저 로드 실패 시 더미 인덱스 사용
                dummy_indices = list(range(len(valid_tokens)))
                print(f"   🔍 Using dummy indices: {dummy_indices}")
                return dummy_indices, valid_tokens
        else:
            print(f"   ⚠️ Tokenizer not found for model {model_key}")
            
            # 토크나이저가 없는 모델의 경우 더미 인덱스 사용
            dummy_indices = list(range(len(valid_tokens)))
            print(f"   🔍 Using dummy indices: {dummy_indices}")
            return dummy_indices, valid_tokens
        
    except Exception as e:
        print(f"❌ Evidence 추출 중 오류: {str(e)}")
        print(f"   프롬프트: {prompt[:100]}...")
        print(f"   모델: {model_key}")
        print(f"   도메인: {domain}")
        return [], []

@st.cache_resource
def load_tokenizer_cached(tokenizer_name: str):
    """토크나이저를 캐시하여 로드합니다."""
    try:
        print(f"   🔧 토크나이저 로드: {tokenizer_name}")
        return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        print(f"   ❌ 토크나이저 로드 실패: {str(e)}")
        # trust_remote_code 없이 다시 시도
        try:
            print(f"   🔧 trust_remote_code=False로 재시도")
            return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
        except Exception as e2:
            print(f"   ❌ 토크나이저 로드 최종 실패: {str(e2)}")
            raise e2

def process_single_prompt_multi_models(prompt_data: Dict[str, Any], model_keys: List[str], domain: str) -> List[Dict[str, Any]]:
    """단일 프롬프트를 여러 모델로 처리합니다."""
    results = []
    
    for model_key in model_keys:
        try:
            print(f"🔍 프롬프트 처리 시작: {domain} 도메인, {model_key} 모델")
            print(f"   프롬프트 길이: {len(prompt_data['prompt'])} 문자")
            print(f"   프롬프트 미리보기: {prompt_data['prompt'][:100]}...")
            
            # 토크나이저 이름 확인
            tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key)
            if not tokenizer_name:
                print(f"   ⚠️ Tokenizer not found for model {model_key}")
                continue
            
            # 토크나이저 로드
            tokenizer = load_tokenizer_cached(tokenizer_name)
            tokens = tokenizer.tokenize(prompt_data["prompt"])
            print(f"   토크나이징 완료: {len(tokens)}개 토큰")
            
            # Evidence 추출
            print(f"   Evidence 추출 시작...")
            evidence_indices, evidence_tokens = extract_evidence_tokens(prompt_data["prompt"], model_key, domain)
            print(f"   Evidence 추출 결과: {len(evidence_indices)}개 인덱스, {len(evidence_tokens)}개 토큰")
            
            # Evidence 추출 결과 확인
            if not evidence_tokens:
                print(f"   ⚠️ No evidence tokens extracted for prompt: {prompt_data['prompt'][:50]}...")
                continue
            
            # response 필드 제거하고 필요한 필드만 포함
            cleaned_data = {
                "prompt": prompt_data["prompt"],
                "domain": prompt_data.get("domain", domain),
                "model": model_key,
                "index": prompt_data.get("index", 0),
                "evidence_indices": evidence_indices,
                "evidence_tokens": evidence_tokens,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            results.append(cleaned_data)
            print(f"   ✅ 프롬프트 처리 완료: {domain} 도메인, {model_key} 모델, {len(evidence_tokens)}개 evidence 토큰")
            
        except Exception as e:
            print(f"   ❌ 프롬프트 처리 중 오류 ({model_key}): {str(e)}")
            print(f"   Prompt: {prompt_data.get('prompt', '')[:50]}...")
            print(f"   Domain: {domain}")
            print(f"   Model: {model_key}")
            continue
    
    return results

def process_single_prompt(prompt_data: Dict[str, Any], model_key: str, domain: str, tokenizer_name: str) -> Dict[str, Any]:
    """단일 프롬프트를 처리합니다. (단일 모델용 - 호환성 유지)"""
    try:
        prompt = prompt_data["prompt"]
        print(f"🔍 프롬프트 처리 시작: {domain} 도메인")
        print(f"   프롬프트 길이: {len(prompt)} 문자")
        print(f"   프롬프트 미리보기: {prompt[:100]}...")
        
        # 토크나이저 로드
        tokenizer = load_tokenizer_cached(tokenizer_name)
        tokens = tokenizer.tokenize(prompt)
        print(f"   토크나이징 완료: {len(tokens)}개 토큰")
        
        # Evidence 추출
        print(f"   Evidence 추출 시작...")
        evidence_indices, evidence_tokens = extract_evidence_tokens(prompt, model_key, domain)
        print(f"   Evidence 추출 결과: {len(evidence_indices)}개 인덱스, {len(evidence_tokens)}개 토큰")
        
        # Evidence 추출 결과 확인
        if not evidence_tokens:
            print(f"   ⚠️ No evidence tokens extracted for prompt: {prompt[:50]}...")
            return None
        
        # response 필드 제거하고 필요한 필드만 포함
        cleaned_data = {
            "prompt": prompt_data["prompt"],
            "domain": prompt_data.get("domain", domain),
            "model": model_key,
            "index": prompt_data.get("index", 0),
            "evidence_indices": evidence_indices,
            "evidence_tokens": evidence_tokens,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        print(f"   ✅ 프롬프트 처리 완료: {len(evidence_tokens)}개 evidence 토큰")
        return cleaned_data
        
    except Exception as e:
        print(f"   ❌ 프롬프트 처리 중 오류: {str(e)}")
        print(f"   Prompt: {prompt_data.get('prompt', '')[:50]}...")
        print(f"   Domain: {domain}")
        print(f"   Model: {model_key}")
        print(f"   Tokenizer: {tokenizer_name}")
        return None

def save_domain_data(domain: str, domain_data: List[Dict[str, Any]], model_key: str, timestamp: str) -> Tuple[Path, int]:
    """도메인 데이터를 파일로 저장합니다."""
    try:
        print(f"💾 저장 시작: {domain} 도메인, {len(domain_data)}개 데이터")
        
        # 모델명에서 콜론을 언더스코어로 변경 (파일 시스템 호환성)
        safe_model_key = model_key.replace(":", "_")
        output_dir = Path(f"dataset/{safe_model_key}/{domain.lower()}")
        
        print(f"📁 출력 디렉토리: {output_dir}")
        
        # 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 생성 완료: {output_dir}")
        
        output_path = output_dir / f"{safe_model_key}_{len(domain_data)}prompts_{timestamp}.jsonl"
        print(f"📄 출력 파일 경로: {output_path}")
        
        # 데이터 검증
        print(f"🔍 데이터 검증:")
        print(f"   - 데이터 타입: {type(domain_data)}")
        print(f"   - 데이터 개수: {len(domain_data)}")
        
        if len(domain_data) > 0:
            print(f"   - 첫 번째 데이터 샘플:")
            first_item = domain_data[0]
            print(f"     - Keys: {list(first_item.keys())}")
            print(f"     - Evidence tokens: {first_item.get('evidence_tokens', 'N/A')}")
            print(f"     - Evidence indices: {first_item.get('evidence_indices', 'N/A')}")
        
        # 파일 저장
        saved_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(domain_data):
                try:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")
                    saved_count += 1
                    
                    # 처음 3개와 마지막 3개만 로그
                    if i < 3 or i >= len(domain_data) - 3:
                        print(f"   ✅ 저장됨 ({i+1}/{len(domain_data)}): {len(item.get('evidence_tokens', []))}개 evidence")
                    
                except Exception as item_error:
                    print(f"   ❌ 항목 {i+1} 저장 실패: {str(item_error)}")
                    print(f"   항목 내용: {item}")
        
        print(f"💾 저장 완료: {saved_count}/{len(domain_data)}개 항목 저장됨")
        print(f"📄 파일 크기: {output_path.stat().st_size if output_path.exists() else 0} bytes")
        
        return output_path, saved_count
        
    except Exception as e:
        print(f"❌ 파일 저장 중 오류 ({domain}): {str(e)}")
        print(f"   - 도메인: {domain}")
        print(f"   - 데이터 개수: {len(domain_data)}")
        print(f"   - 모델: {model_key}")
        print(f"   - 타임스탬프: {timestamp}")
        return None, 0

def show():
    st.title("🔍 Evidence Extractor")
    st.markdown("도메인별 프롬프트 파일에서 evidence 토큰을 추출합니다.")
    
    # 강제 캐시 무효화
    if st.sidebar.button("🔄 강제 새로고침", key="force_refresh_evidence"):
        get_available_models.clear()
        st.success("캐시가 새로고침되었습니다!")
    
    # ===== 섹션 1: 모델 설정 =====
    st.markdown("---")
    st.markdown("## 🤖 Model Configuration")
    
    # 실험 모드 선택
    experiment_mode = st.radio(
        "Evidence 추출 모드를 선택하세요",
        ["단일 모델 추출", "다중 모델 추출"],
        key="evidence_mode_selector"
    )
    
    # 모델 선택
    col1, col2 = st.columns([2, 1])
    
    with col1:
        available_models = get_available_models()
        if not available_models:
            st.error("❌ Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.")
            return
        
        if experiment_mode == "단일 모델 추출":
            model_key = st.selectbox(
                "모델 선택",
                available_models,
                index=0 if available_models else None,
                help="Evidence 추출에 사용할 모델을 선택하세요."
            )
            selected_models = [model_key]
        else:
            st.markdown("**🔧 추출할 모델들을 선택하세요 (여러 개 선택 가능)**")
            selected_models = st.multiselect(
                "사용 가능한 모델들",
                available_models,
                default=[available_models[0]] if available_models else [],
                help="여러 모델을 선택하면 순차적으로 처리됩니다."
            )
            model_key = selected_models[0] if selected_models else None
    
    with col2:
        if st.button("🔄 모델 목록 새로고침", key="refresh_models_evidence"):
            get_available_models.clear()
            st.success("모델 목록이 새로고침되었습니다!")
            st.rerun()
    
    # 모델 실행 상태 확인
    st.markdown("### 🔍 Model Status Check")
    
    # 실행 중인 모델 목록 표시
    running_models = get_running_models()
    col_status1, col_status2 = st.columns([1, 2])
    
    with col_status1:
        if st.button("🔄 상태 새로고침", key="refresh_status"):
            st.rerun()
    
    with col_status2:
        if running_models:
            st.success(f"🟢 실행 중인 모델: {', '.join(running_models)}")
        else:
            st.warning("⚪ 실행 중인 모델이 없습니다.")
    
    # 선택된 모델들 상태 확인
    if selected_models:
        st.markdown("**📊 선택된 모델 상태 확인**")
        
        for model in selected_models:
            is_running = model in running_models
            is_available = check_model_status(model)
            
            col_status3, col_status4 = st.columns(2)
            
            with col_status3:
                if is_running:
                    st.success(f"✅ {model} - 실행 중")
                elif is_available:
                    st.info(f"ℹ️ {model} - 사용 가능 (로드 필요)")
                else:
                    st.error(f"❌ {model} - 사용 불가")
            
            with col_status4:
                # 토크나이저 이름 확인
                tokenizer_name = MODEL_TOKENIZER_MAP.get(model)
                if tokenizer_name:
                    st.success(f"🔧 {model} - 토크나이저 지원")
                else:
                    st.warning(f"⚠️ {model} - 토크나이저 없음")
        
        # 모든 모델이 사용 가능한지 확인
        all_available = all(check_model_status(model) for model in selected_models)
        if not all_available:
            st.warning("⚠️ 일부 모델을 먼저 로드해주세요.")
            st.markdown("""
            **해결 방법:**
            1. Model Loader 탭으로 이동
            2. 모델을 선택하고 "🚀 Start Model" 클릭
            3. 또는 터미널에서: `ollama run {model_key}`
            """)
            return
        
        st.success(f"✅ {len(selected_models)}개 모델 설정이 완료되었습니다!")
    else:
        st.warning("⚠️ 최소 하나의 모델을 선택해주세요.")
        return
    
    # ===== 섹션 2: 도메인 선택 =====
    st.markdown("---")
    st.markdown("## 📁 Domain Selection")
    
    # 사용 가능한 도메인 확인
    origin_dir = Path("dataset/origin")
    if not origin_dir.exists():
        st.error("❌ `dataset/origin` 디렉토리가 존재하지 않습니다. 먼저 도메인 프롬프트를 생성해주세요.")
        return
    
    available_domains = [d.name for d in origin_dir.iterdir() if d.is_dir()]
    if not available_domains:
        st.error("❌ 사용 가능한 도메인이 없습니다. 먼저 도메인 프롬프트를 생성해주세요.")
        return
    
    # 도메인 선택
    col3, col4 = st.columns([2, 1])
    
    with col3:
        selected_domains = st.multiselect(
            "도메인 선택",
            available_domains,
            default=available_domains[:2] if len(available_domains) >= 2 else available_domains,
            help="Evidence를 추출할 도메인을 선택하세요. 여러 개 선택 가능합니다."
        )
    
    with col4:
        show_info = st.button("📊 도메인 정보 보기", key="show_domain_info")
    
    # 도메인 정보 표시
    if show_info:
        st.markdown("---")
        st.subheader("📊 Domain Information")
        
        for domain in available_domains:
            prompts = load_origin_prompts(domain)
            col5, col6 = st.columns([3, 1])
            with col5:
                st.write(f"📄 **{domain}** 도메인")
            with col6:
                st.write(f"📝 {len(prompts)}개 프롬프트")
        
        st.info("💡 도메인 정보를 다시 보려면 '📊 도메인 정보 보기' 버튼을 클릭하세요.")
    
    # ===== 섹션 3: 미리보기 기능 =====
    st.markdown("---")
    st.markdown("## 👀 Preview Evidence Extraction")
    
    preview_enabled = st.checkbox("미리보기 활성화", value=True, help="실제 추출 전에 샘플로 테스트해볼 수 있습니다.")
    
    if preview_enabled and selected_domains:
        preview_domain = st.selectbox("미리보기할 도메인 선택", selected_domains)
        preview_prompts = load_origin_prompts(preview_domain)
        
        if preview_prompts:
            preview_index = st.slider("미리보기할 프롬프트 인덱스", 0, min(len(preview_prompts)-1, 10), 0)
            
            if experiment_mode == "단일 모델 추출":
                # 단일 모델 미리보기
                if st.button("🔍 미리보기 실행", key="preview_evidence"):
                    with st.spinner("미리보기 중..."):
                        preview_prompt_data = preview_prompts[preview_index]
                        preview_prompt = preview_prompt_data["prompt"]
                        
                        # 토크나이징
                        tokenizer = load_tokenizer_cached(tokenizer_name)
                        tokens = tokenizer.tokenize(preview_prompt)
                        
                        # Evidence 추출
                        evidence_indices, evidence_tokens = extract_evidence_tokens(
                            preview_prompt, model_key, preview_domain
                        )
            else:
                # 다중 모델 미리보기
                preview_models = st.multiselect(
                    "미리보기할 모델 선택",
                    selected_models,
                    default=selected_models[:2] if len(selected_models) >= 2 else selected_models,
                    help="미리보기할 모델들을 선택하세요."
                )
                
                if st.button("🔍 다중 모델 미리보기 실행", key="preview_evidence_multi"):
                    with st.spinner("다중 모델 미리보기 중..."):
                        preview_prompt_data = preview_prompts[preview_index]
                        preview_prompt = preview_prompt_data["prompt"]
                        
                        # 다중 모델로 처리
                        processed_items = process_single_prompt_multi_models(
                            preview_prompt_data, preview_models, preview_domain
                        )
                        
                        if processed_items:
                            st.markdown("### 📋 Multi-Model Preview Results")
                            
                            for item in processed_items:
                                model_name = item['model']
                                evidence_tokens = item['evidence_tokens']
                                evidence_indices = item['evidence_indices']
                                
                                st.markdown(f"**🔧 {model_name} 모델 결과:**")
                                
                                col7, col8 = st.columns(2)
                                with col7:
                                    st.write(f"**추출된 Evidence 토큰 ({len(evidence_tokens)}개):**")
                                    st.text_area("", str(evidence_tokens), height=100, key=f"preview_evidence_tokens_{model_name}")
                                
                                with col8:
                                    st.write(f"**Evidence 토큰 위치 ({len(evidence_indices)}개):**")
                                    st.text_area("", str(evidence_indices), height=100, key=f"preview_evidence_indices_{model_name}")
                                
                                st.markdown("---")
                        
                        # 원본 프롬프트 표시
                        st.markdown("### 📄 Original Prompt")
                        st.text_area("", preview_prompt, height=100, key="preview_prompt_multi")
                        
                        # 토크나이징 결과 (첫 번째 모델 기준)
                        if preview_models:
                            first_model = preview_models[0]
                            tokenizer_name = MODEL_TOKENIZER_MAP.get(first_model)
                            if tokenizer_name:
                                tokenizer = load_tokenizer_cached(tokenizer_name)
                                tokens = tokenizer.tokenize(preview_prompt)
                                st.write(f"**토크나이징 결과 (참고용, {first_model} 기준):**")
                                st.write(f"총 {len(tokens)}개 토큰")
                                st.text_area("", str(tokens[:20]) + "..." if len(tokens) > 20 else str(tokens), height=100, key="preview_tokens_multi")
        
        # 단일 모델 미리보기 결과 표시
        if experiment_mode == "단일 모델 추출" and 'preview_prompt' in locals():
            # 결과 표시
            st.markdown("### 📋 Preview Results")
            
            col7, col8 = st.columns(2)
            with col7:
                st.write("**원본 프롬프트:**")
                st.text_area("", preview_prompt, height=100, key="preview_prompt")
                
                st.write("**토크나이징 결과 (참고용):**")
                st.write(f"총 {len(tokens)}개 토큰")
                st.text_area("", str(tokens[:20]) + "..." if len(tokens) > 20 else str(tokens), height=100, key="preview_tokens")
            
            with col8:
                st.write("**추출된 Evidence 토큰:**")
                st.write(f"총 {len(evidence_tokens)}개 토큰")
                st.text_area("", str(evidence_tokens), height=100, key="preview_evidence_tokens")
                
                st.write("**Evidence 토큰 위치 (원본 프롬프트 기준):**")
                st.write(f"총 {len(evidence_indices)}개 위치")
                st.text_area("", str(evidence_indices), height=100, key="preview_evidence_indices")
                
                # 디버깅 정보 추가
                st.markdown("### 🔍 Debug Information")
                
                # LLM 응답 확인
                domain_instruction = DOMAIN_PROMPTS.get(preview_domain.lower(), "Find important tokens related to the domain.")
                debug_prompt = f"""
You are an English-only evidence extraction system. Your task is to extract English tokens from the given prompt.

DOMAIN: {preview_domain}
INSTRUCTION: {domain_instruction}

INPUT PROMPT: "{preview_prompt}"

CRITICAL RULES:
1. You MUST respond ONLY in English
2. You MUST extract ONLY English words that exist in the input prompt
3. You MUST return ONLY a JSON array format
4. You MUST NOT translate words
5. You MUST NOT add words that are not in the prompt
6. You MUST NOT respond in Korean or any other language
7. You MUST focus on domain-specific medical/technical terms

EXAMPLES:
- For medical domain: ["clinical", "findings", "diagnosis", "viral", "encephalitis", "adults"]
- For legal domain: ["legal", "contract", "liability", "jurisdiction"]
- For technical domain: ["algorithm", "implementation", "optimization"]

RESPONSE FORMAT (JSON array only):
["word1", "word2", "word3"]
"""
                
                debug_response = call_ollama_api(model_key, debug_prompt)
                
                col9, col10 = st.columns(2)
                with col9:
                    st.write("**LLM 응답 (원본):**")
                    st.text_area("", debug_response or "No response", height=150, key="debug_response")
                    
                    st.write("**추출된 토큰 (파싱 후):**")
                    parsed_tokens = extract_tokens_from_response(debug_response) if debug_response else []
                    st.text_area("", str(parsed_tokens), height=100, key="debug_parsed_tokens")
                
                with col10:
                    st.write("**토큰 매칭 결과:**")
                    prompt_lower = preview_prompt.lower()
                    matching_info = []
                    for token in parsed_tokens or []:
                        token_lower = token.lower().strip()
                        is_in_prompt = token_lower in prompt_lower
                        import re
                        pattern = r'\b' + re.escape(token_lower) + r'\b'
                        is_word_boundary = bool(re.search(pattern, prompt_lower))
                        matching_info.append(f"'{token}': in_prompt={is_in_prompt}, word_boundary={is_word_boundary}")
                    
                    st.text_area("", "\n".join(matching_info), height=150, key="debug_matching")
                
                # 인덱스 계산 디버깅 정보 추가
                st.markdown("### 🔍 Index Calculation Debug")
                
                col11, col12 = st.columns(2)
                with col11:
                    st.write("**토큰화 결과 (인덱스 표시):**")
                    tokenized = tokenizer.tokenize(preview_prompt)
                    indexed_tokens = ""
                    for i, token in enumerate(tokenized):
                        if i % 5 == 0:
                            indexed_tokens += f"\n{i:3d}: "
                        indexed_tokens += f"{token} "
                    st.text_area("", indexed_tokens, height=200, key="indexed_tokens")
                
                with col12:
                    st.write("**Evidence 토큰 위치 확인 (토큰 단위):**")
                    position_info = []
                    
                    # 원본 evidence 토큰과 인덱스 매칭
                    original_tokens = extract_tokens_from_response(debug_response) if debug_response else []
                    english_tokens = []
                    for token in original_tokens or []:
                        token_clean = token.lower().strip()
                        if token_clean and all(c.isascii() and c.isalnum() or c.isspace() for c in token_clean):
                            english_tokens.append(token_clean)
                    
                    # 각 토큰에 대한 인덱스 찾기
                    all_indices = find_token_positions(preview_prompt, english_tokens, tokenizer)
                    
                    for i, (token, idx) in enumerate(zip(english_tokens, all_indices)):
                        if idx != -1 and idx < len(tokenized):
                            actual_token = tokenized[idx]
                            # Ġ 문자 제거하여 표시
                            clean_actual_token = actual_token.replace('Ġ', '')
                            position_info.append(f"'{token}' at token pos {idx}: '{clean_actual_token}' ✅")
                        elif idx == -1:
                            position_info.append(f"'{token}': not found in tokens ❌")
                        else:
                            position_info.append(f"'{token}' at token pos {idx}: index out of range ❌")
                    
                    st.text_area("", "\n".join(position_info), height=200, key="position_info")
                
                # 최종 데이터셋 항목 미리보기
                st.markdown("### 📊 Final Dataset Item Preview")
                final_item = {
                    "prompt": preview_prompt_data["prompt"],
                    "domain": preview_prompt_data.get("domain", preview_domain),
                    "model": model_key,
                    "index": preview_prompt_data.get("index", preview_index),
                    "evidence_indices": evidence_indices,
                    "evidence_tokens": evidence_tokens,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                st.json(final_item)
        else:
            st.warning(f"⚠️ {preview_domain} 도메인에 프롬프트가 없습니다.")
    
    # ===== 섹션 4: Evidence 추출 실행 =====
    st.markdown("---")
    st.markdown("## ⚙️ Extraction Settings")
    
    # Evidence 추출 설정
    col9, col10 = st.columns([1, 2])
    
    with col9:
        batch_size = st.number_input(
            "배치 크기",
            min_value=1,
            max_value=50,
            value=10,
            help="한 번에 처리할 프롬프트 개수입니다."
        )
        
        # 테스트 모드 옵션 추가
        test_mode = st.checkbox(
            "테스트 모드 (처음 5개만 처리)",
            value=False,
            help="처음 5개의 프롬프트만 처리하여 테스트합니다."
        )
    
    with col10:
        st.info("✅ 선택된 도메인의 모든 프롬프트에서 evidence 토큰을 추출합니다.")
    
    # Evidence 추출 버튼
    col11, col12 = st.columns([2, 1])
    
    with col11:
        extract_button = st.button("🔍 Extract Evidence", type="primary", key="extract_evidence")
    
    with col12:
        if st.button("🗑️ 캐시 초기화", key="clear_cache_evidence"):
            get_available_models.clear()
            load_tokenizer_cached.clear()  # 토크나이저 캐시도 초기화
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith('tokenizer_')]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("캐시가 초기화되었습니다!")
    
    # ===== Evidence 추출 실행 =====
    if extract_button and selected_domains:
        try:
            # 사전 체크
            st.markdown("### 🔍 Pre-Extraction Check")
            
            # 1. 모델 상태 확인
            if experiment_mode == "단일 모델 추출":
                # 단일 모델 체크
                is_model_running = model_key in get_running_models()
                if is_model_running:
                    st.success(f"✅ 모델 {model_key} 실행 중")
                else:
                    st.error(f"❌ 모델 {model_key} 실행되지 않음")
                    st.warning("💡 Model Loader 탭에서 모델을 먼저 실행해주세요.")
                    return
                
                # 토크나이저 확인
                tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key)
                if tokenizer_name:
                    st.success(f"✅ 토크나이저: {tokenizer_name}")
                else:
                    st.error(f"❌ 토크나이저를 찾을 수 없음: {model_key}")
                    return
            else:
                # 다중 모델 체크
                for model in selected_models:
                    is_model_running = model in get_running_models()
                    if is_model_running:
                        st.success(f"✅ 모델 {model} 실행 중")
                    else:
                        st.warning(f"⚠️ 모델 {model} 실행되지 않음 (자동 로드 시도)")
                    
                    # 토크나이저 확인
                    tokenizer_name = MODEL_TOKENIZER_MAP.get(model)
                    if tokenizer_name:
                        st.success(f"✅ {model} 토크나이저: {tokenizer_name}")
                    else:
                        st.warning(f"⚠️ {model} 토크나이저 없음")
            
            # 2. 도메인별 프롬프트 확인
            for domain in selected_domains:
                prompts = load_origin_prompts(domain)
                if prompts:
                    st.success(f"✅ {domain} 도메인: {len(prompts)}개 프롬프트")
                else:
                    st.error(f"❌ {domain} 도메인: 프롬프트 없음")
                    return
            
            st.success("✅ 모든 사전 체크 통과! Evidence 추출을 시작합니다.")
            st.markdown("---")
            
            # 전체 시작 시간 기록
            total_start_time = time.time()
            
            # 진행상황 표시를 위한 컨테이너들
            progress_text = st.empty()
            progress_counter = st.empty()
            time_info = st.empty()
            
            # 최종 데이터셋을 저장할 리스트
            final_datasets = []
            
            # 전체 작업량 계산
            total_tasks = len(selected_domains) * len(selected_models)
            completed_tasks = 0
            
            for domain_idx, domain in enumerate(selected_domains, 1):
                # 도메인별 시작 시간 기록
                domain_start_time = time.time()
                
                # 도메인 프롬프트 로드
                prompts = load_origin_prompts(domain)
                if not prompts:
                    st.warning(f"⚠️ {domain} 도메인에 프롬프트가 없습니다.")
                    continue
                
                # 테스트 모드인 경우 처음 5개만 처리
                if test_mode:
                    prompts = prompts[:5]
                    st.info(f"🧪 테스트 모드: {domain} 도메인에서 처음 5개 프롬프트만 처리합니다.")
                
                if experiment_mode == "단일 모델 추출":
                    # 단일 모델 처리
                    progress_text.text(f"Extracting evidence from {domain} domain prompts using {model_key}...")
                    
                    with st.spinner(f"Extracting evidence from {len(prompts)} prompts in {domain} domain using {model_key} ({domain_idx}/{len(selected_domains)})..."):
                        for i, prompt_data in enumerate(prompts):
                            # 진행상황 카운터 업데이트
                            progress_counter.text(f"{domain} domain: {i+1}/{len(prompts)}")
                            
                            # 시간 정보 업데이트
                            current_time = time.time()
                            elapsed_time = current_time - total_start_time
                            
                            # 예상 시간 계산
                            total_prompts_to_process = sum(len(load_origin_prompts(d)) for d in selected_domains)
                            completed_prompts = sum(len(load_origin_prompts(d)) for d in selected_domains[:domain_idx-1]) + i
                            if completed_prompts > 0:
                                avg_time_per_prompt = elapsed_time / completed_prompts
                                remaining_prompts = total_prompts_to_process - completed_prompts
                                estimated_remaining_time = avg_time_per_prompt * remaining_prompts
                                estimated_total_time = elapsed_time + estimated_remaining_time
                                
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: {estimated_total_time:.1f}초 | 남은시간: {estimated_remaining_time:.1f}초")
                            else:
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: 계산 중...")
                            
                            # 단일 프롬프트 처리
                            processed_item = process_single_prompt(prompt_data, model_key, domain, tokenizer_name)
                            if processed_item:
                                final_datasets.append(processed_item)
                                # 디버깅: 성공한 경우 로그
                                if (i + 1) % 10 == 0:
                                    print(f"✅ Processed {i+1}/{len(prompts)} prompts in {domain} domain - Evidence tokens: {len(processed_item.get('evidence_tokens', []))}")
                            else:
                                # 디버깅: 실패한 경우 상세 로그
                                print(f"❌ Failed to process prompt {i+1} in {domain} domain")
                                if i < 3:  # 처음 3개만 상세 로그
                                    print(f"   Prompt: {prompt_data.get('prompt', '')[:100]}...")
                                    print(f"   Domain: {domain}")
                                    print(f"   Model: {model_key}")
                                    print(f"   Tokenizer: {tokenizer_name}")
                                if (i + 1) % 10 == 0:
                                    print(f"❌ Failed {i+1}/{len(prompts)} prompts in {domain} domain")
                        
                        # 도메인별 완료 시간 계산
                        domain_end_time = time.time()
                        domain_duration = domain_end_time - domain_start_time
                        print(f"{domain} domain evidence extraction completed in {domain_duration:.2f} seconds")
                
                else:
                    # 다중 모델 처리
                    progress_text.text(f"Extracting evidence from {domain} domain prompts using {len(selected_models)} models...")
                    
                    with st.spinner(f"Extracting evidence from {len(prompts)} prompts in {domain} domain using {len(selected_models)} models ({domain_idx}/{len(selected_domains)})..."):
                        for i, prompt_data in enumerate(prompts):
                            # 진행상황 카운터 업데이트
                            progress_counter.text(f"{domain} domain: {i+1}/{len(prompts)} (모델: {len(selected_models)}개)")
                            
                            # 시간 정보 업데이트
                            current_time = time.time()
                            elapsed_time = current_time - total_start_time
                            
                            # 예상 시간 계산 (모델 수를 고려)
                            total_prompts_to_process = sum(len(load_origin_prompts(d)) for d in selected_domains) * len(selected_models)
                            completed_prompts = (sum(len(load_origin_prompts(d)) for d in selected_domains[:domain_idx-1]) + i) * len(selected_models)
                            if completed_prompts > 0:
                                avg_time_per_prompt = elapsed_time / completed_prompts
                                remaining_prompts = total_prompts_to_process - completed_prompts
                                estimated_remaining_time = avg_time_per_prompt * remaining_prompts
                                estimated_total_time = elapsed_time + estimated_remaining_time
                                
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: {estimated_total_time:.1f}초 | 남은시간: {estimated_remaining_time:.1f}초")
                            else:
                                time_info.text(f"소요시간: {elapsed_time:.1f}초 | 예상완료: 계산 중...")
                            
                            # 다중 모델로 프롬프트 처리
                            processed_items = process_single_prompt_multi_models(prompt_data, selected_models, domain)
                            if processed_items:
                                final_datasets.extend(processed_items)
                                # 디버깅: 성공한 경우 로그
                                if (i + 1) % 10 == 0:
                                    total_evidence = sum(len(item.get('evidence_tokens', [])) for item in processed_items)
                                    print(f"✅ Processed {i+1}/{len(prompts)} prompts in {domain} domain with {len(selected_models)} models - Total evidence tokens: {total_evidence}")
                            else:
                                # 디버깅: 실패한 경우 상세 로그
                                print(f"❌ Failed to process prompt {i+1} in {domain} domain with all models")
                                if i < 3:  # 처음 3개만 상세 로그
                                    print(f"   Prompt: {prompt_data.get('prompt', '')[:100]}...")
                                    print(f"   Domain: {domain}")
                                    print(f"   Models: {selected_models}")
                                if (i + 1) % 10 == 0:
                                    print(f"❌ Failed {i+1}/{len(prompts)} prompts in {domain} domain")
                        
                        # 도메인별 완료 시간 계산
                        domain_end_time = time.time()
                        domain_duration = domain_end_time - domain_start_time
                        print(f"{domain} domain evidence extraction completed in {domain_duration:.2f} seconds")
            
            # 전체 완료 시간 계산
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            
            # 진행상황 표시 정리
            progress_text.empty()
            progress_counter.empty()
            time_info.empty()
            
            # ===== 최종 데이터셋 저장 =====
            st.markdown("---")
            st.subheader("💾 Saving Final Dataset")
            
            # 디버깅 정보 표시
            st.info(f"📊 총 처리된 데이터: {len(final_datasets)}개")
            st.info(f"🎯 선택된 도메인: {', '.join(selected_domains)}")
            
            # final_datasets 상세 분석
            print(f"🔍 Final datasets 분석:")
            print(f"   - 총 데이터 개수: {len(final_datasets)}")
            
            if len(final_datasets) > 0:
                print(f"   - 첫 번째 데이터 샘플:")
                first_data = final_datasets[0]
                print(f"     - Keys: {list(first_data.keys())}")
                print(f"     - Domain: {first_data.get('domain', 'N/A')}")
                print(f"     - Model: {first_data.get('model', 'N/A')}")
                print(f"     - Evidence tokens: {first_data.get('evidence_tokens', [])}")
                print(f"     - Evidence indices: {first_data.get('evidence_indices', [])}")
            
            # 도메인별 데이터 분포 확인
            domain_distribution = {}
            for item in final_datasets:
                domain = item.get('domain', 'unknown')
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            
            print(f"   - 도메인별 분포: {domain_distribution}")
            
            # 각 도메인별 처리 결과 표시
            for domain in selected_domains:
                # 대소문자 구분 없이 매칭
                domain_data = [item for item in final_datasets if item["domain"].lower() == domain.lower()]
                if domain_data:
                    st.success(f"✅ {domain} 도메인: {len(domain_data)}개 데이터 처리 완료")
                    print(f"✅ {domain} 도메인: {len(domain_data)}개 데이터 확인됨")
                else:
                    st.error(f"❌ {domain} 도메인: 데이터 처리 실패")
                    print(f"❌ {domain} 도메인: 데이터 없음")
                    # 실패 원인 분석
                    prompts = load_origin_prompts(domain)
                    if prompts:
                        st.warning(f"   - 원본 프롬프트: {len(prompts)}개 존재")
                        st.warning(f"   - 모델 상태: {'실행 중' if model_key in get_running_models() else '실행되지 않음'}")
                        st.warning(f"   - 토크나이저: {MODEL_TOKENIZER_MAP.get(model_key, '없음')}")
            
            # 도메인별로 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            print(f"💾 저장 시작 - 타임스탬프: {timestamp}")
            
            if experiment_mode == "단일 모델 추출":
                # 단일 모델 저장
                for domain in selected_domains:
                    # 도메인별 데이터 필터링 (대소문자 구분 없이)
                    domain_data = [item for item in final_datasets if item["domain"].lower() == domain.lower()]
                    
                    st.info(f"📋 {domain} 도메인 데이터: {len(domain_data)}개")
                    print(f"📋 {domain} 도메인 저장 준비: {len(domain_data)}개 데이터")
                    
                    if domain_data:
                        output_path, count = save_domain_data(domain, domain_data, model_key, timestamp)
                        if output_path:
                            saved_files.append((domain, output_path, count))
                            st.success(f"✅ {domain} 도메인 파일 저장 완료: {output_path}")
                            print(f"✅ {domain} 도메인 저장 성공: {output_path} ({count}개)")
                        else:
                            st.error(f"❌ {domain} 도메인 파일 저장 실패")
                            print(f"❌ {domain} 도메인 저장 실패")
                    else:
                        st.warning(f"⚠️ {domain} 도메인에 저장할 데이터가 없습니다.")
                        print(f"⚠️ {domain} 도메인: 저장할 데이터 없음")
                        st.info(f"💡 원인: evidence 추출 과정에서 모든 프롬프트가 실패했을 수 있습니다.")
                        st.info(f"💡 해결: 미리보기 기능으로 개별 프롬프트를 테스트해보세요.")
            else:
                # 다중 모델 저장 - 모델별로 분리하여 저장
                for model in selected_models:
                    st.markdown(f"**📊 {model} 모델 결과 저장**")
                    
                    for domain in selected_domains:
                        # 모델과 도메인별 데이터 필터링
                        domain_data = [item for item in final_datasets 
                                     if item["domain"].lower() == domain.lower() and item["model"] == model]
                        
                        st.info(f"📋 {domain} 도메인 ({model}): {len(domain_data)}개")
                        print(f"📋 {domain} 도메인 ({model}) 저장 준비: {len(domain_data)}개 데이터")
                        
                        if domain_data:
                            output_path, count = save_domain_data(domain, domain_data, model, timestamp)
                            if output_path:
                                saved_files.append((f"{domain} ({model})", output_path, count))
                                st.success(f"✅ {domain} 도메인 ({model}) 파일 저장 완료: {output_path}")
                                print(f"✅ {domain} 도메인 ({model}) 저장 성공: {output_path} ({count}개)")
                            else:
                                st.error(f"❌ {domain} 도메인 ({model}) 파일 저장 실패")
                                print(f"❌ {domain} 도메인 ({model}) 저장 실패")
                        else:
                            st.warning(f"⚠️ {domain} 도메인 ({model})에 저장할 데이터가 없습니다.")
                            print(f"⚠️ {domain} 도메인 ({model}): 저장할 데이터 없음")
            
            # ===== 최종 결과 표시 =====
            st.markdown("---")
            st.subheader("✅ Evidence Extraction Complete!")
            
            # 결과 요약을 컬럼으로 배치
            col13, col14, col15 = st.columns(3)
            
            with col13:
                st.metric("총 소요 시간", f"{total_duration:.1f}초")
            
            with col14:
                if experiment_mode == "단일 모델 추출":
                    st.metric("처리된 도메인", f"{len(selected_domains)}개")
                else:
                    st.metric("처리된 도메인", f"{len(selected_domains)}개")
            
            with col15:
                total_generated = len(final_datasets)
                if experiment_mode == "단일 모델 추출":
                    st.metric("생성된 데이터셋", f"{total_generated}개")
                else:
                    st.metric("생성된 데이터셋", f"{total_generated}개")
            
            # 다중 모델 결과 요약
            if experiment_mode == "다중 모델 추출":
                st.markdown("**📊 모델별 결과 요약**")
                
                model_summary = {}
                for item in final_datasets:
                    model_name = item['model']
                    if model_name not in model_summary:
                        model_summary[model_name] = {'count': 0, 'domains': set()}
                    model_summary[model_name]['count'] += 1
                    model_summary[model_name]['domains'].add(item['domain'])
                
                for model_name, summary in model_summary.items():
                    domains_str = ', '.join(sorted(summary['domains']))
                    st.info(f"**{model_name}**: {summary['count']}개 결과 ({domains_str} 도메인)")
            
            # 생성된 파일 목록 표시
            st.markdown("---")
            st.subheader("📋 Generated Files")
            
            if saved_files:
                for domain, file_path, count in saved_files:
                    with st.container():
                        col16, col17 = st.columns([3, 1])
                        with col16:
                            # 파일이 실제로 존재하는지 확인
                            if file_path.exists():
                                st.write(f"📄 **{domain}** 도메인: `{file_path.name}` ({count}개 항목) ✅")
                            else:
                                st.write(f"📄 **{domain}** 도메인: `{file_path.name}` ({count}개 항목) ❌ 파일 없음")
                        with col17:
                            st.write(f"📍 `{file_path.parent}`")
                
                # 출력 디렉토리 정보
                if experiment_mode == "단일 모델 추출":
                    safe_model_key = model_key.replace(":", "_")
                    st.info(f"📁 모든 파일이 `dataset/{safe_model_key}/` 디렉토리에 저장되었습니다.")
                else:
                    st.info(f"📁 각 모델별로 `dataset/[모델명]/` 디렉토리에 저장되었습니다.")
                    for model in selected_models:
                        safe_model_key = model.replace(":", "_")
                        st.info(f"   - {model}: `dataset/{safe_model_key}/`")
                
                # 파일 크기 정보 표시
                st.markdown("---")
                st.subheader("📏 File Information")
                for domain, file_path, count in saved_files:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        st.write(f"📄 {domain}: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            else:
                st.warning("⚠️ 저장된 파일이 없습니다.")
                st.info("💡 Evidence 추출이 완료되지 않았거나 저장 중 오류가 발생했을 수 있습니다.")
            
        except Exception as e:
            st.session_state.evidence_extraction_error = str(e)
            print(f"Evidence extraction error: {str(e)}")
    
    # Evidence 추출 실패 시 에러 처리
    if extract_button and not selected_domains:
        st.error("❌ 도메인을 선택해주세요!")
    
    # Evidence 추출 중 에러 처리
    if 'evidence_extraction_error' in st.session_state:
        st.markdown("---")
        st.subheader("❌ Evidence Extraction Failed")
        
        # 에러 정보를 컬럼으로 배치
        col18, col19 = st.columns([2, 1])
        
        with col18:
            st.error(f"Evidence 추출 중 오류가 발생했습니다:")
            st.code(st.session_state.evidence_extraction_error)
        
        with col19:
            st.warning("💡 해결 방법:")
            st.write("1. 토크나이저가 올바른지 확인")
            st.write("2. 모델이 실행 중인지 확인")
            st.write("3. 프롬프트 파일이 존재하는지 확인")
        
        # 에러 상태 정리
        del st.session_state.evidence_extraction_error 