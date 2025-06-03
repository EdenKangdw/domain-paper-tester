import streamlit as st
from pathlib import Path
import json
from transformers import AutoTokenizer
from utils import check_ollama_model_status, OLLAMA_API_BASE
import requests

def get_running_models():
    """현재 실행 중인 Ollama 모델 목록 가져오기"""
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
    """모델의 응답 가져오기"""
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
        return response.json()["response"]
    except Exception as e:
        return f"오류 발생: {str(e)}"

def tokenize_and_extract_words(text, tokenizer):
    """텍스트를 토큰화하고 단어를 추출"""
    tokens = tokenizer.tokenize(text)
    words = []
    word_to_tokens = {}
    current_word = []
    current_word_tokens = []
    
    for i, token in enumerate(tokens):
        if token.startswith('▁'):
            if current_word:
                word = ''.join(current_word).replace('▁', '')
                if word.strip():  # 빈 문자열이 아닌 경우만 추가
                    words.append(word)
                    word_to_tokens[word] = current_word_tokens
                current_word = []
                current_word_tokens = []
        current_word.append(token)
        current_word_tokens.append(i)
    
    if current_word:
        word = ''.join(current_word).replace('▁', '')
        if word.strip():  # 빈 문자열이 아닌 경우만 추가
            words.append(word)
            word_to_tokens[word] = current_word_tokens
    
    # 중복 단어 제거하면서 순서 유지
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)
    
    return tokens, unique_words, word_to_tokens

def format_word_and_token_info(tokens, words, word_to_tokens):
    """토큰과 단어 정보를 포맷팅"""
    token_entries = [
        f'토큰[{i}] = >>>{token}<<< (원문자: {repr(token)})'
        for i, token in enumerate(tokens)
    ]
    
    word_entries = [
        f'단어[{i}] = >>>{word}<<<'
        for i, word in enumerate(words)
    ]
    
    return "\n".join(token_entries), "\n".join(word_entries)

def create_evidence_query(word_list, prompt, domain):
    """Evidence 추출을 위한 쿼리 생성"""
    return f"""아래 단어 목록에서만 '{domain}' 분야와 직접적으로 관련된 단어들을 찾아주세요.
프롬프트에 없는 단어나 목록에 없는 단어는 절대 포함하지 마세요.

프롬프트: "{prompt}"

선택 가능한 단어 목록 (이 목록에 있는 단어만 선택 가능):
{word_list}

주의사항:
- 위 단어 목록에 있는 단어만 선택할 수 있습니다
- 목록에 없는 단어는 절대로 포함하지 마세요
- 단어를 수정하거나 변형하지 마세요
- 새로운 단어를 만들지 마세요
- 도메인과 관련된 단어가 없다면 빈 배열을 반환하세요

응답 규칙:
1. evidence_word_index에는 선택한 단어의 번호만 넣으세요 (단어 목록의 번호)
2. evidence에는 선택한 단어를 정확하게 복사해서 넣으세요
3. 관련 단어가 없다면 빈 배열을 반환하세요
4. 목록에 없는 단어는 절대 포함하지 마세요

응답 형식:
{{
    "evidence_word_index": [단어 번호1, 단어 번호2, ...],
    "evidence": ["단어1", "단어2", ...],
    "explanation": "선택한 단어들이 {domain} 분야와 관련된 이유를 설명해주세요. 관련 단어가 없다면 '관련 단어를 찾을 수 없습니다.'라고 작성하세요."
}}

검증 사항:
1. evidence_word_index의 각 번호는 0부터 {len(word_list.split(chr(10))) - 1} 사이의 값이어야 합니다
2. evidence의 각 단어는 반드시 단어 목록에 있어야 합니다
3. 단어는 >>> <<< 사이의 내용을 정확하게 복사해야 합니다
4. 새로운 단어를 만들거나 기존 단어를 수정하지 마세요"""

def extract_json_from_response(response):
    """응답에서 JSON 추출"""
    import re
    json_match = re.search(r'(\{[^{]*\})', response)
    if not json_match:
        raise ValueError("JSON 형식의 응답을 찾을 수 없습니다")
    return json.loads(json_match.group(1))

def validate_evidence(result, words):
    """Evidence 결과 검증"""
    required_fields = ["evidence_word_index", "evidence", "explanation"]
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise ValueError(f"누락된 필드가 있습니다: {', '.join(missing_fields)}")
    
    evidence_word_index = result["evidence_word_index"]
    evidence = result["evidence"]
    
    if not isinstance(evidence_word_index, list):
        raise ValueError("evidence_word_index는 배열([]) 형식이어야 합니다")
    if not isinstance(evidence, list):
        raise ValueError("evidence는 배열([]) 형식이어야 합니다")
    
    # 인덱스 유효성 검사
    invalid_indices = []
    for i, idx in enumerate(evidence_word_index):
        if not isinstance(idx, int):
            invalid_indices.append({"위치": i, "인덱스": idx, "이유": "정수가 아님"})
        elif not (0 <= idx < len(words)):
            invalid_indices.append({"위치": i, "인덱스": idx, "이유": f"범위 초과 (0-{len(words)-1})"})
    
    if invalid_indices:
        details = [
            f"위치 {e['위치']}: 인덱스 {e['인덱스']} ({e['이유']})"
            for e in invalid_indices
        ]
        raise ValueError(f"유효하지 않은 인덱스가 있습니다:\n" + "\n".join(details))
    
    # evidence와 evidence_word_index 길이 일치 검사
    if len(evidence) != len(evidence_word_index):
        raise ValueError(f"배열 길이가 일치하지 않습니다 (evidence: {len(evidence)}, index: {len(evidence_word_index)})")
    
    # 단어 목록에 없는 단어 검사
    invalid_words = []
    for i, word in enumerate(evidence):
        if word not in words:
            invalid_words.append({
                "위치": i,
                "단어": word,
                "가능한_단어": words
            })
    
    if invalid_words:
        details = [
            f"위치 {w['위치']}: '{w['단어']}' (선택 가능한 단어: {w['가능한_단어']})"
            for w in invalid_words
        ]
        raise ValueError(f"단어 목록에 없는 단어가 포함되어 있습니다:\n" + "\n".join(details))
    
    # 인덱스와 단어 매칭 검사
    mismatches = []
    for i, (idx, word) in enumerate(zip(evidence_word_index, evidence)):
        if words[idx] != word:
            mismatches.append({
                "위치": i,
                "인덱스": idx,
                "예상": words[idx],
                "실제": word
            })
    
    if mismatches:
        details = [
            f"위치 {m['위치']}: 인덱스 {m['인덱스']}는 '{m['예상']}'이지만 '{m['실제']}'가 사용됨"
            for m in mismatches
        ]
        raise ValueError(f"인덱스와 단어가 일치하지 않습니다:\n" + "\n".join(details))
    
    return evidence_word_index, evidence

def visualize_evidence(words, evidence_word_index, evidence, explanation):
    """Evidence 결과 시각화"""
    highlighted_words = [
        f"<span style='background-color:#fff176; padding:2px'>{word}</span>"
        if i in evidence_word_index else word
        for i, word in enumerate(words)
    ]
    
    st.markdown("### 추출된 Evidence:")
    st.markdown(" ".join(highlighted_words), unsafe_allow_html=True)
    st.json({
        "evidence_word_index": evidence_word_index,
        "evidence": evidence,
        "explanation": explanation
    })

def show():
    st.title("🧪 Evidence 추출 + 저장 (Ollama 기반)")

    # 현재 실행 중인 모델 표시
    running_models = get_running_models()
    if running_models:
        st.success(f"✅ 현재 실행 중인 모델: {', '.join(running_models)}")
    else:
        st.warning("⚠️ 현재 실행 중인 모델이 없습니다. 모델 로드 탭에서 모델을 시작해주세요.")
        st.stop()

    domains = ["의료", "법률", "보험", "금융", "회계"]
    models = running_models if running_models else ["llama2", "gemma", "qwen", "deepseek"]
    
    # 모델별 토크나이저 설정
    MODEL_TOKENIZER_MAP = {
        "llama2": "meta-llama/Llama-2-7b-hf",
        "gemma:2b": "google/gemma-2b",
        "gemma:7b": "google/gemma-7b",
        "qwen": "Qwen/Qwen-7B",
        "deepseek": "deepseek-ai/deepseek-coder-7b-base"
    }

    # 입력 섹션
    st.subheader("📝 입력")
    prompt = st.text_area("프롬프트를 입력하세요", height=100)
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("도메인을 선택하세요", domains)
    with col2:
        selected_model = st.selectbox(
            "사용할 Ollama 모델을 선택하세요", 
            models,
            help="실행 중인 모델 중에서 선택할 수 있습니다."
        )
    
    # 선택한 모델에 맞는 토크나이저 로드
    tokenizer = None
    if prompt.strip():
        try:
            # 모델 이름에서 버전 정보 추출
            model_key = selected_model.lower()
            tokenizer_name = MODEL_TOKENIZER_MAP.get(model_key)
            
            if not tokenizer_name:
                # 버전 정보가 없는 기본 모델 이름으로 다시 시도
                base_model = model_key.split(":")[0]
                tokenizer_name = MODEL_TOKENIZER_MAP.get(base_model)
            
            if tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                tokens = tokenizer.tokenize(prompt)
                if st.button("🔍 토크나이저 결과 보기", key="show_tokenizer_result"):
                    st.markdown("### 토크나이저 결과")
                    # 토큰과 인덱스를 함께 표시
                    token_data = [{"인덱스": i, "토큰": token} for i, token in enumerate(tokens)]
                    st.table(token_data)
            else:
                st.warning(f"⚠️ {selected_model} 모델의 토크나이저를 찾을 수 없습니다. 지원되는 모델: {', '.join(MODEL_TOKENIZER_MAP.keys())}")
                st.stop()
        except Exception as e:
            st.error(f"❌ 토크나이저 로드 중 오류 발생: {str(e)}")
            st.stop()

    # 프리뷰 섹션
    st.subheader("👀 프리뷰")
    if prompt.strip() and tokenizer:
        if st.button("🎯 Evidence 추출 미리보기", key="show_evidence_preview"):
            with st.spinner("Evidence 추출 중..."):
                # 토큰화 및 단어 추출
                tokens, words, word_to_tokens = tokenize_and_extract_words(prompt, tokenizer)
                
                # 정보 포맷팅
                token_list, word_list = format_word_and_token_info(tokens, words, word_to_tokens)
                
                # 쿼리 생성 및 모델 호출
                query = create_evidence_query(word_list, prompt, domain)
                evidence_response = get_model_response(selected_model, query)
                
                try:
                    # JSON 추출 및 검증
                    result = extract_json_from_response(evidence_response)
                    evidence_word_index, evidence = validate_evidence(result, words)
                    explanation = result.get("explanation", "")
                    
                    # 결과 시각화
                    visualize_evidence(words, evidence_word_index, evidence, explanation)
                    
                except Exception as e:
                    st.error(str(e))
                    st.code(evidence_response, language="text")

    # 저장 섹션
    st.subheader("💾 저장")
    if st.button("📦 Evidence 추출 결과 저장"):
        if not prompt.strip():
            st.warning("프롬프트를 입력해주세요.")
        else:
            # 선택한 모델이 실행 중인지 다시 한번 확인
            if not check_ollama_model_status(selected_model):
                st.error(f"❌ {selected_model} 모델이 실행되고 있지 않습니다. 모델 로드 탭에서 모델을 시작해주세요.")
                st.stop()

            try:
                with st.spinner("Evidence 추출 및 저장 중..."):
                    # 일반 응답 얻기
                    response = get_model_response(selected_model, prompt)

                    # Evidence 추출
                    query = f"""입력된 프롬프트에서 '{domain}' 분야와 관련된 단어들을 찾아주세요.

프롬프트: "{prompt}"

단어 목록:
{word_list}

토큰 정보:
{token_list}

주의사항:
- 프롬프트 내에서만 단어를 찾으세요
- 도메인과 관련된 단어가 없다면 빈 배열을 반환하세요
- 단어는 정확히 제시된 형태로만 사용해야 합니다
- 단어를 수정하거나 변형하지 마세요
- evidence 배열의 각 단어는 단어 목록에서 복사한 것과 정확히 일치해야 합니다

응답 규칙:
1. 프롬프트 내에서 '{domain}' 분야와 직접적으로 관련된 단어만 찾으세요
2. 관련 단어가 없다면 빈 배열을 반환하세요
3. evidence_word_index에는 선택한 단어의 번호만 넣으세요
4. evidence에는 해당 번호의 단어를 정확하게 복사해서 넣으세요
5. evidence_word_index와 evidence 배열의 길이는 같아야 합니다

응답 형식:
{{
    "evidence_word_index": [단어 번호1, 단어 번호2, ...],
    "evidence": ["단어1", "단어2", ...],
    "explanation": "선택한 단어들이 {domain} 분야와 관련된 이유를 설명해주세요. 관련 단어가 없다면 '관련 단어를 찾을 수 없습니다.'라고 작성하세요."
}}

검증 사항:
1. evidence_word_index의 각 번호는 실제 단어 목록의 인덱스여야 합니다
2. evidence의 각 단어는 해당 인덱스의 단어와 정확히 일치해야 합니다
3. 단어는 >>> <<< 사이의 내용을 그대로 복사해야 합니다
4. 도메인과 관련 없는 단어는 포함하지 마세요"""

                    evidence_response = get_model_response(selected_model, query)
                    try:
                        # 응답에서 JSON 부분만 추출
                        import re
                        json_match = re.search(r'(\{[^{]*\})', evidence_response)
                        if not json_match:
                            raise ValueError("JSON 형식의 응답을 찾을 수 없습니다")
                        
                        evidence_response = json_match.group(1)
                        result = json.loads(evidence_response)
                        
                        # 필수 필드 검증
                        required_fields = ["evidence_word_index", "evidence", "explanation"]
                        missing_fields = [field for field in required_fields if field not in result]
                        if missing_fields:
                            raise ValueError(f"누락된 필드가 있습니다: {', '.join(missing_fields)}")
                            
                        evidence_word_index = result["evidence_word_index"]
                        evidence = result["evidence"]
                        explanation = result.get("explanation", "")

                        # 리스트 형식 검증
                        if not isinstance(evidence_word_index, list):
                            raise ValueError("evidence_word_index는 배열([]) 형식이어야 합니다")
                        if not isinstance(evidence, list):
                            raise ValueError("evidence는 배열([]) 형식이어야 합니다")

                        # 인덱스 유효성 검사
                        invalid_indices = [i for i in evidence_word_index if not (isinstance(i, int) and 0 <= i < len(words))]
                        if invalid_indices:
                            raise ValueError(f"유효하지 않은 인덱스가 있습니다: {invalid_indices}")

                        # evidence와 evidence_word_index 길이 일치 검사
                        if len(evidence) != len(evidence_word_index):
                            raise ValueError(f"배열 길이가 일치하지 않습니다 (evidence: {len(evidence)}, index: {len(evidence_word_index)})")

                        # evidence가 실제 단어와 일치하는지 검사
                        mismatches = []
                        for i, idx in enumerate(evidence_word_index):
                            expected_word = words[idx]
                            actual_word = evidence[i]
                            if expected_word != actual_word:
                                mismatches.append({
                                    "위치": i,
                                    "인덱스": idx,
                                    "예상": repr(expected_word),
                                    "실제": repr(actual_word)
                                })
                        
                        if mismatches:
                            mismatch_details = [
                                f"위치 {m['위치']}: 인덱스 {m['인덱스']}의 단어가 일치하지 않음 (예상: {m['예상']}, 실제: {m['실제']})"
                                for m in mismatches
                            ]
                            raise ValueError(f"단어 불일치:\n" + "\n".join(mismatch_details))

                        # 저장
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
                        output_path = output_dir / f"{selected_model}_{domain}.jsonl"
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(output, ensure_ascii=False) + "\n")

                        # 결과 표시
                        st.success(f"🎉 저장 완료: {output_path}")
                        
                        # 저장된 결과 미리보기
                        with st.expander("📋 저장된 결과 보기"):
                            st.markdown("### 모델 응답:")
                            st.markdown(response)
                            
                            st.markdown("### 추출된 Evidence:")
                            # 단어 단위로 결과 표시
                            word_results = []
                            for i, word in enumerate(words):
                                is_evidence = i in evidence_word_index
                                word_results.append({
                                    "인덱스": i,
                                    "단어": word,
                                    "Evidence 여부": "✅" if is_evidence else ""
                                })
                            st.table(word_results)
                            
                            st.markdown("### Evidence 설명:")
                            st.markdown(explanation)
                            
                            st.markdown("### 전체 결과:")
                            st.json({
                                "evidence_word_index": evidence_word_index,
                                "evidence": evidence,
                                "explanation": explanation
                            })

                    except json.JSONDecodeError as e:
                        st.error(f"Evidence 추출에 실패했습니다. JSON 파싱 오류: {str(e)}")
                        st.code(evidence_response, language="text")
                    except ValueError as e:
                        st.error(f"Evidence 추출에 실패했습니다. 데이터 검증 오류: {str(e)}")
                        st.code(evidence_response, language="text")
                    except Exception as e:
                        st.error(f"Evidence 추출 중 오류가 발생했습니다: {str(e)}")
                        st.code(evidence_response, language="text")

            except Exception as e:
                st.error(f"❌ Ollama 요청 실패: {e}") 