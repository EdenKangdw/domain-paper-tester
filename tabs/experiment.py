import streamlit as st
import os
import json
from utils import get_available_models
import pandas as pd
from datetime import datetime
import pickle

# Huggingface 관련 추가
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_ROOT = "dataset"
MODEL_CACHE_FILE = "model_cache.pkl"

# 모델별 실험 결과 저장 경로
def get_model_experiment_path(model_name):
    """모델별 실험 결과 저장 경로를 반환합니다."""
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    return f"experiment_results/{safe_model_name}"

def get_model_dataset_path(model_name):
    """모델별 데이터셋 저장 경로를 반환합니다."""
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    return f"dataset/{safe_model_name}"

# 글로벌 모델 캐시
MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None
}

def save_model_cache():
    """모델 캐시 정보를 파일에 저장"""
    try:
        cache_info = {
            "model_name": MODEL_CACHE["model_name"],
            "timestamp": datetime.now().isoformat()
        }
        with open(MODEL_CACHE_FILE, "wb") as f:
            pickle.dump(cache_info, f)
    except Exception as e:
        print(f"캐시 저장 실패: {e}")

def load_model_cache():
    """파일에서 모델 캐시 정보를 로드"""
    try:
        if os.path.exists(MODEL_CACHE_FILE):
            with open(MODEL_CACHE_FILE, "rb") as f:
                cache_info = pickle.load(f)
            return cache_info
    except Exception as e:
        print(f"캐시 로드 실패: {e}")
    return None

def clear_model_cache():
    """모델 캐시 파일 삭제"""
    try:
        if os.path.exists(MODEL_CACHE_FILE):
            os.remove(MODEL_CACHE_FILE)
    except Exception as e:
        print(f"캐시 삭제 실패: {e}")

def create_model_directories(model_name):
    """모델별 디렉토리를 생성합니다."""
    experiment_path = get_model_experiment_path(model_name)
    dataset_path = get_model_dataset_path(model_name)
    
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(dataset_path, exist_ok=True)
    
    # 도메인별 하위 디렉토리 생성
    domains = ["general", "technical", "legal", "medical"]
    for domain in domains:
        os.makedirs(os.path.join(dataset_path, domain), exist_ok=True)
    
    return experiment_path, dataset_path

def get_model_dataset_files(model_name):
    """특정 모델의 데이터셋 파일 목록을 가져옵니다."""
    dataset_path = get_model_dataset_path(model_name)
    domains = ["general", "technical", "legal", "medical"]
    files = {}
    
    for domain in domains:
        domain_path = os.path.join(dataset_path, domain)
        if os.path.exists(domain_path):
            files[domain] = [f for f in os.listdir(domain_path) if f.endswith(".jsonl")]
        else:
            files[domain] = []
    
    return files

def get_model_prompts(model_name, domain, filename, max_count=10000):
    """특정 모델의 데이터셋에서 프롬프트 리스트를 추출합니다."""
    dataset_path = get_model_dataset_path(model_name)
    path = os.path.join(dataset_path, domain, filename)
    prompts = []
    
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_count:
                    break
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt", "(no prompt)")
                    prompts.append(prompt)
                except Exception:
                    continue
    except Exception:
        pass
    return prompts

def copy_original_dataset_to_model(model_name):
    """원본 데이터셋을 모델별 디렉토리로 복사합니다."""
    original_dataset_path = DATASET_ROOT
    model_dataset_path = get_model_dataset_path(model_name)
    
    if not os.path.exists(original_dataset_path):
        st.error("원본 데이터셋이 존재하지 않습니다.")
        return False
    
    try:
        # 원본 데이터셋의 모든 파일을 모델별 디렉토리로 복사
        domains = ["general", "technical", "legal", "medical"]
        for domain in domains:
            original_domain_path = os.path.join(original_dataset_path, domain)
            model_domain_path = os.path.join(model_dataset_path, domain)
            
            if os.path.exists(original_domain_path):
                # 도메인 디렉토리 생성
                os.makedirs(model_domain_path, exist_ok=True)
                
                # 파일 복사
                for filename in os.listdir(original_domain_path):
                    if filename.endswith('.jsonl'):
                        original_file = os.path.join(original_domain_path, filename)
                        model_file = os.path.join(model_domain_path, filename)
                        
                        # 파일이 없으면 복사
                        if not os.path.exists(model_file):
                            import shutil
                            shutil.copy2(original_file, model_file)
        
        st.success(f"{model_name} 모델용 데이터셋이 준비되었습니다.")
        return True
    except Exception as e:
        st.error(f"데이터셋 복사 중 오류 발생: {str(e)}")
        return False

# 각 도메인별 데이터셋 파일 목록을 가져오는 함수 (기존 호환성 유지)
def get_dataset_files():
    domains = ["general", "technical", "legal", "medical"]
    files = {}
    for domain in domains:
        domain_path = os.path.join(DATASET_ROOT, domain)
        if os.path.exists(domain_path):
            files[domain] = [f for f in os.listdir(domain_path) if f.endswith(".jsonl")]
        else:
            files[domain] = []
    return files

# 선택한 데이터셋 파일에서 프롬프트 리스트를 추출하는 함수 (기존 호환성 유지)
def get_prompts(domain, filename, max_count=10000):
    path = os.path.join(DATASET_ROOT, domain, filename)
    prompts = []
    try:
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= max_count:
                    break
                try:
                    data = json.loads(line)
                    prompt = data.get("prompt", "(no prompt)")
                    prompts.append(prompt)
                except Exception:
                    continue
    except Exception:
        pass
    return prompts

def load_model_to_session(model_name):
    """
    모델을 서버 메모리에 로드하고 세션 상태와 글로벌 캐시에 참조를 저장합니다.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    model_map = {
        'mistral:7b': 'mistralai/Mistral-7B-v0.1',
        'llama2:7b': 'meta-llama/Llama-2-7b-hf',
        'gemma:7b': 'google/gemma-7b',
        'gemma:2b': 'google/gemma-2b',
        'qwen:7b': 'Qwen/Qwen-7B',
        'qwen:14b': 'Qwen/Qwen-14B',
        'deepseek:7b': 'deepseek-ai/deepseek-llm-7b-base',
        'yi:6b': '01-ai/Yi-6B',
        'yi:34b': '01-ai/Yi-34B',
        'openchat:7b': 'openchat/openchat-3.5-7b',
        'neural:7b': 'microsoft/DialoGPT-medium',
        'phi:2.7b': 'microsoft/phi-2',
        'stable:7b': 'stabilityai/stablelm-base-alpha-7b',
    }
    hf_model = model_map.get(model_name)
    if not hf_model:
        st.error("지원하지 않는 모델명입니다.")
        return False
    try:
        # 모델과 토크나이저를 서버 메모리에 로드
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        
        # 모델별 특별한 토크나이저 설정
        if 'gemma' in model_name.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        elif 'llama' in model_name.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        elif 'qwen' in model_name.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        
        # 모델 로딩 시도
        try:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        except Exception as quant_error:
            # 8비트 양자화 실패 시 16비트로 시도
            st.warning("8비트 양자화 로딩 실패, 16비트로 시도합니다...")
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        # 세션 상태와 글로벌 캐시에 참조 저장
        st.session_state['model'] = model
        st.session_state['tokenizer'] = tokenizer
        st.session_state['model_name'] = model_name
        MODEL_CACHE["model"] = model
        MODEL_CACHE["tokenizer"] = tokenizer
        MODEL_CACHE["model_name"] = model_name
        # 캐시 정보를 파일에 저장
        save_model_cache()
        return True
    except Exception as e:
        st.error(f"모델 로딩 실패: {str(e)}")
        st.error("모델명을 확인하거나 인터넷 연결을 확인해주세요.")
        return False

def unload_model_from_session():
    """
    서버 메모리에서 모델을 해제합니다.
    세션 상태와 글로벌 캐시의 참조를 제거하고 GPU 메모리를 비웁니다.
    """
    import torch
    if 'model' in st.session_state:
        del st.session_state['model']
    if 'tokenizer' in st.session_state:
        del st.session_state['tokenizer']
    if 'model_name' in st.session_state:
        del st.session_state['model_name']
    # 글로벌 캐시도 비움
    MODEL_CACHE["model"] = None
    MODEL_CACHE["tokenizer"] = None
    MODEL_CACHE["model_name"] = None
    # 캐시 파일 삭제
    clear_model_cache()
    # GPU 메모리 정리
    torch.cuda.empty_cache()
    st.success("모델이 서버 메모리에서 해제되었습니다.")

def get_attention_from_session(prompt):
    import torch
    model = st.session_state.get('model', None)
    tokenizer = st.session_state.get('tokenizer', None)
    if model is None or tokenizer is None:
        return None, None, None
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = tuple(attn.cpu().numpy() for attn in outputs.attentions)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_ids = inputs['input_ids'][0].cpu().tolist()
    return attentions, tokens, token_ids

def plot_attention_head_heatmap(attn, tokens, evidence_indices):
    """
    attn: (head, from_token, to_token) numpy array (마지막 레이어)
    tokens: 토큰 리스트
    evidence_indices: evidence 토큰의 인덱스 리스트
    """
    # evidence_indices 유효성 검사 및 필터링
    if not evidence_indices:
        evidence_indices = []
    else:
        # 토큰 길이를 벗어나는 인덱스 제거
        evidence_indices = [i for i in evidence_indices if 0 <= i < len(tokens)]
    
    # 각 헤드별로 evidence 토큰(to_token)에 대한 어텐션 평균 계산
    # attn shape: (head, from_token, to_token)
    head_count = attn.shape[0]
    avg_evidence_attention = []
    for h in range(head_count):
        # from_token 전체에서 evidence 토큰(to_token)으로 가는 어텐션 평균
        if evidence_indices:  # evidence_indices가 비어있지 않을 때만 계산
            try:
                avg = attn[h, :, evidence_indices].mean()
            except (IndexError, ValueError):
                avg = 0.0  # 인덱스 에러 발생 시 0 반환
        else:
            avg = 0.0  # evidence_indices가 비어있으면 0 반환
        avg_evidence_attention.append(avg)
    avg_evidence_attention = np.array(avg_evidence_attention)

    # 히트맵 시각화
    fig, ax = plt.subplots(figsize=(10, 1.5))
    sns.heatmap(avg_evidence_attention[None, :], annot=True, fmt=".2f", cmap="YlOrRd", cbar=True, ax=ax)
    ax.set_xlabel("Head index")
    ax.set_yticks([])
    ax.set_title("Average Evidence Attention by Head")
    plt.tight_layout()
    return fig, avg_evidence_attention

def plot_attention_head_token_heatmap(attn, tokens, evidence_indices):
    """
    attn: (head, from_token, to_token) numpy array (마지막 레이어)
    tokens: 토큰 리스트
    evidence_indices: evidence 토큰의 인덱스 리스트
    """
    # evidence_indices 유효성 검사 및 필터링
    if not evidence_indices:
        evidence_indices = []
    else:
        # 토큰 길이를 벗어나는 인덱스 제거
        evidence_indices = [i for i in evidence_indices if 0 <= i < len(tokens)]
    
    # 각 헤드별로 to_token(모든 토큰)에 대한 어텐션 평균 (from_token 전체 평균)
    # shape: (head, to_token)
    avg_attn = attn.mean(axis=1)  # (head, to_token)

    fig, ax = plt.subplots(figsize=(min(1.2+0.4*len(tokens), 16), 1.5+0.3*avg_attn.shape[0]))
    sns.heatmap(avg_attn, annot=False, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_xlabel("Token index")
    ax.set_ylabel("Head index")
    ax.set_title("Average Token Attention by Head")
    # x축에 토큰 라벨 표시 (길면 잘림)
    token_labels = [t if i not in evidence_indices else f"*{t}*" for i, t in enumerate(tokens)]
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
    # evidence 토큰 강조 (x축 라벨 색상)
    for idx in evidence_indices:
        if idx < len(ax.get_xticklabels()):  # 인덱스 범위 확인
            ax.get_xticklabels()[idx].set_color("red")
            ax.get_xticklabels()[idx].set_fontweight("bold")
    plt.tight_layout()
    return fig, avg_attn

def plot_token_head_attention_heatmap(attn, tokens, evidence_indices):
    """
    attn: (head, from_token, to_token) numpy array (마지막 레이어)
    tokens: 토큰 리스트
    evidence_indices: evidence 토큰의 인덱스 리스트
    """
    # evidence_indices 유효성 검사 및 필터링
    if not evidence_indices:
        evidence_indices = []
    else:
        # 토큰 길이를 벗어나는 인덱스 제거
        evidence_indices = [i for i in evidence_indices if 0 <= i < len(tokens)]
    
    # 각 토큰별로 헤드 어텐션 평균 (from_token 전체 평균)
    # shape: (head, to_token)
    avg_attn = attn.mean(axis=1)  # (head, to_token)
    # shape를 (to_token, head)로 transpose
    avg_attn_t = avg_attn.T  # (to_token, head)
    fig, ax = plt.subplots(figsize=(min(1.2+0.4*len(tokens), 16), 1.5+0.3*avg_attn_t.shape[1]))
    sns.heatmap(avg_attn_t, annot=False, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_ylabel("Token index")
    ax.set_xlabel("Head index")
    ax.set_title("Average Head Attention by Token")
    # y축에 토큰 라벨 표시 (길면 잘림)
    token_labels = [t if i not in evidence_indices else f"*{t}*" for i, t in enumerate(tokens)]
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(token_labels, rotation=0, fontsize=8)
    # evidence 토큰 강조 (y축 라벨 색상)
    for idx in evidence_indices:
        if idx < len(ax.get_yticklabels()):  # 인덱스 범위 확인
            ax.get_yticklabels()[idx].set_color("red")
            ax.get_yticklabels()[idx].set_fontweight("bold")
    plt.tight_layout()
    return fig, avg_attn_t

def batch_domain_experiment(model_name, files, num_prompts=5):
    """
    여러 도메인에 대해 evidence 어텐션 실험을 일괄 수행하고 통계 집계
    model_name: 사용할 모델명
    files: get_dataset_files() 결과
    num_prompts: 도메인별 샘플링할 프롬프트 개수
    """
    results = []
    
    # 서버 메모리에 로드된 모델이 있는지 확인
    if 'model' in st.session_state and 'tokenizer' in st.session_state:
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        model_loaded_in_session = True
    else:
        st.error("모델이 세션에 로드되어 있지 않습니다. 실험을 진행하려면 먼저 모델을 로드해주세요.")
        return results

    for domain, file_list in files.items():
        if not file_list:
            continue
        selected_file = file_list[0]  # 각 도메인 첫 번째 파일 사용(확장 가능)
        # 모델별 데이터셋에서 프롬프트 가져오기
        prompts = get_model_prompts(model_name, domain, selected_file, max_count=10000)
        if not prompts:
            continue
        # 프롬프트 샘플링
        sampled = prompts[:num_prompts]
        # 모델별 데이터셋 경로 사용
        path = os.path.join(get_model_dataset_path(model_name), domain, selected_file)
        with open(path, "r") as f:
            lines = f.readlines()
        for prompt in sampled:
            try:
                idx = prompts.index(prompt)
                data = json.loads(lines[idx])
                evidence_indices = data.get("evidence_indices", [])
                # 어텐션 추출
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
                attentions = tuple(attn.cpu().numpy() for attn in outputs.attentions)
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                evidence_indices = [i for i in evidence_indices if i < len(tokens)]
                last_attn = attentions[-1][0]  # (head, from_token, to_token)
                # 헤드별 evidence 어텐션 평균
                head_count = last_attn.shape[0]
                avg_evidence_attention = []
                for h in range(head_count):
                    if evidence_indices:  # evidence_indices가 비어있지 않을 때만 계산
                        try:
                            avg = last_attn[h, :, evidence_indices].mean()
                        except (IndexError, ValueError):
                            avg = 0.0  # 인덱스 에러 발생 시 0 반환
                    else:
                        avg = 0.0  # evidence_indices가 비어있으면 0 반환
                    avg_evidence_attention.append(avg)
                max_head = int(np.argmax(avg_evidence_attention))
                results.append({
                    "domain": domain,
                    "prompt": prompt,
                    "max_head": max_head,
                    "avg_evidence_attention": avg_evidence_attention[max_head],
                    "evidence_indices": evidence_indices,
                    "tokens": tokens,
                    "model_name": model_name,
                    "tokenizer_name": tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "unknown"
                })
            except Exception as e:
                continue

    return results

def save_experiment_result(results, model_name):
    """실험 결과를 모델별 experiment_results 폴더에 저장 (numpy 타입을 모두 변환)"""
    # 모델별 디렉토리 생성
    experiment_path = get_model_experiment_path(model_name)
    os.makedirs(experiment_path, exist_ok=True)
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_path}/{now}_{model_name.replace(':','_')}.json"
    
    # numpy 타입을 모두 파이썬 기본 타입으로 변환
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, (list, tuple)):
            return [convert(x) for x in o]
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        return o
    results_converted = [convert(r) for r in results]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results_converted, f, ensure_ascii=False, indent=2)
    return filename

def list_model_experiment_results(model_name):
    """특정 모델의 실험 결과 파일 목록을 반환합니다."""
    experiment_path = get_model_experiment_path(model_name)
    if not os.path.exists(experiment_path):
        return []
    
    files = [f for f in os.listdir(experiment_path) if f.endswith(".json")]
    files.sort(reverse=True)
    return files

def load_model_experiment_result(model_name, filename):
    """특정 모델의 실험 결과를 로드합니다."""
    experiment_path = get_model_experiment_path(model_name)
    path = os.path.join(experiment_path, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_all_model_experiments():
    """모든 모델의 실험 결과를 조회합니다."""
    experiment_root = "experiment_results"
    if not os.path.exists(experiment_root):
        return {}
    
    model_experiments = {}
    for model_dir in os.listdir(experiment_root):
        model_path = os.path.join(experiment_root, model_dir)
        if os.path.isdir(model_path):
            # 디렉토리명을 모델명으로 변환
            model_name = model_dir.replace('_', '/').replace('_', ':')
            files = [f for f in os.listdir(model_path) if f.endswith(".json")]
            if files:
                model_experiments[model_name] = files
    
    return model_experiments

def check_model_loaded():
    """
    서버 메모리에 모델이 로드되어 있는지 확인합니다.
    세션에 없으면 글로벌 캐시에서 복구하고, 그것도 없으면 파일 캐시에서 복구합니다.
    """
    try:
        # 1. 먼저 세션 상태 확인
        if 'model' in st.session_state and 'tokenizer' in st.session_state:
            model = st.session_state['model']
            if model is not None:
                return True, st.session_state.get('model_name', '알 수 없는 모델')
        
        # 2. 세션에 없으면 글로벌 캐시에서 복구
        if MODEL_CACHE["model"] is not None and MODEL_CACHE["tokenizer"] is not None:
            st.session_state['model'] = MODEL_CACHE["model"]
            st.session_state['tokenizer'] = MODEL_CACHE["tokenizer"]
            st.session_state['model_name'] = MODEL_CACHE["model_name"]
            return True, MODEL_CACHE["model_name"]
        
        # 3. 글로벌 캐시에도 없으면 파일 캐시에서 복구 시도
        cache_info = load_model_cache()
        if cache_info and cache_info.get('model_name'):
            model_name = cache_info['model_name']
            # 모델을 다시 로드
            if load_model_to_session(model_name):
                return True, model_name
        
        # 4. 세션에 모델 이름만 있고 실제 모델이 없는 경우 (새로고침 후)
        if 'model_name' in st.session_state and st.session_state['model_name']:
            model_name = st.session_state['model_name']
            # 모델을 다시 로드
            if load_model_to_session(model_name):
                return True, model_name
        
        return False, None
    except Exception as e:
        print(f"모델 체크 중 에러 발생: {str(e)}")
        return False, None

def show():
    # 페이지 로드 시 자동으로 모델 상태 복구 시도
    if 'model_auto_restored' not in st.session_state:
        st.session_state['model_auto_restored'] = True
        # 자동 복구 시도 (조용히)
        try:
            is_loaded, loaded_model_name = check_model_loaded()
            if is_loaded:
                st.session_state['auto_restored_model'] = loaded_model_name
        except Exception:
            pass
    
    st.title("실험 탭")
    st.write("실험 탭입니다. (어텐션 히트맵 및 evidence 분석 기능이 여기에 추가됨)")

    # 자동 복구된 모델이 있으면 알림
    if 'auto_restored_model' in st.session_state:
        st.success(f"이전에 로드된 {st.session_state['auto_restored_model']} 모델이 자동으로 복구되었습니다.")
        del st.session_state['auto_restored_model']

    st.markdown("---")
    st.subheader(":rocket: huggingface 모델 로드/해제")
    
    # 모델 로드 상태 확인
    is_loaded, loaded_model_name = check_model_loaded()
    if is_loaded:
        st.success(f"현재 {loaded_model_name} 모델이 서버 메모리에 로드되어 있습니다.")
    
    # Hugging Face 모델 목록 (실험용)
    hf_model_list = [
        "mistral:7b",
        "llama2:7b", 
        "gemma:7b",
        "gemma:2b",
        "qwen:7b",
        "qwen:14b",
        "deepseek:7b",
        "yi:6b",
        "yi:34b",
        "openchat:7b",
        "neural:7b",
        "phi:2.7b",
        "stable:7b"
    ]
    
    # Ollama 모델 목록 (참고용)
    ollama_model_list = get_available_models()
    
    # 모델 선택 방식
    model_selection_method = st.radio(
        "모델 선택 방식",
        ["Hugging Face 모델 (실험용)", "Ollama 모델 (참고용)", "직접 입력"],
        horizontal=True,
        key="model_selection_method"
    )
    
    if model_selection_method == "Hugging Face 모델 (실험용)":
        selected_model = st.selectbox(
            "사용할 Hugging Face 모델을 선택하세요", 
            hf_model_list, 
            key="hf_model_select"
        )
        st.info("💡 Hugging Face 모델은 어텐션 실험에 사용됩니다.")
        
    elif model_selection_method == "Ollama 모델 (참고용)":
        if ollama_model_list:
            selected_model = st.selectbox(
                "설치된 Ollama 모델 목록 (참고용)", 
                ollama_model_list, 
                key="ollama_model_select"
            )
            st.warning("⚠️ Ollama 모델은 어텐션 실험을 지원하지 않습니다. Hugging Face 모델을 사용해주세요.")
        else:
            selected_model = st.text_input("설치된 Ollama 모델이 없습니다. 직접 입력하세요", key="ollama_model_input")
            st.warning("⚠️ Ollama 모델은 어텐션 실험을 지원하지 않습니다.")
    else:
        selected_model = st.text_input("모델명을 직접 입력하세요 (예: mistral:7b)", key="custom_model_input")
        st.info("💡 지원되는 모델: mistral:7b, llama2:7b, gemma:7b, qwen:7b 등")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("모델 로드"):
            if is_loaded:
                st.warning(f"이미 {loaded_model_name} 모델이 로드되어 있습니다. 다른 모델을 로드하려면 먼저 현재 모델을 해제해주세요.")
            else:
                if load_model_to_session(selected_model):
                    st.success(f"모델 {selected_model}이(가) 서버 메모리에 로드되었습니다.")
                    st.rerun()  # 페이지 새로고침하여 상태 업데이트
    with col2:
        if st.button("모델 해제"):
            if not is_loaded:
                st.warning("로드된 모델이 없습니다.")
            else:
                unload_model_from_session()
                st.rerun()  # 페이지 새로고침하여 상태 업데이트

    # 모델이 로드되어 있으면 실험 UI
    if is_loaded:
        st.markdown("---")
        st.subheader(":gear: 모델별 실험 설정")
        
        # 모델별 데이터셋 준비
        with st.expander("📁 모델별 데이터셋 준비", expanded=False):
            st.markdown("""
            **모델별 실험을 위해 데이터셋을 준비합니다:**
            - 각 모델마다 별도의 데이터셋 디렉토리가 생성됩니다.
            - 원본 데이터셋이 모델별 디렉토리로 복사됩니다.
            - 모델별로 독립적인 실험 환경을 구성할 수 있습니다.
            """)
            
            if st.button("현재 모델용 데이터셋 준비"):
                with st.spinner(f"{loaded_model_name} 모델용 데이터셋을 준비하는 중..."):
                    if copy_original_dataset_to_model(loaded_model_name):
                        st.success(f"{loaded_model_name} 모델용 데이터셋이 준비되었습니다.")
                        st.rerun()
        
        # 모델별 데이터셋 확인
        model_dataset_files = get_model_dataset_files(loaded_model_name)
        has_model_dataset = any(files for files in model_dataset_files.values())
        
        if has_model_dataset:
            st.success(f"✅ {loaded_model_name} 모델용 데이터셋이 준비되어 있습니다.")
            
            # 모델별 실험 UI
            st.markdown("---")
            st.subheader(":microscope: 모델별 실험")
            
            domains = list(model_dataset_files.keys())
            selected_domain = st.selectbox("도메인 선택", domains, key="model_domain")
            dataset_files = model_dataset_files[selected_domain]
            
            if dataset_files:
                selected_file = st.selectbox("데이터셋 파일 선택", dataset_files, key="model_file")
                prompts = get_model_prompts(loaded_model_name, selected_domain, selected_file)
                
                if prompts:
                    selected_prompt = st.selectbox("프롬프트 선택 (최대 100개 미리보기)", prompts, key="model_prompt")
                    prompt_idx = prompts.index(selected_prompt)
                    path = os.path.join(get_model_dataset_path(loaded_model_name), selected_domain, selected_file)
                    
                    # 선택된 프롬프트의 전체 데이터 로드
                    try:
                        with open(path, "r") as f:
                            for i, line in enumerate(f):
                                if i == prompt_idx:
                                    data = json.loads(line)
                                    # 전체 데이터를 보기 좋게 표시
                                    st.markdown("### 📝 선택된 데이터셋 상세 정보")
                                    st.markdown("**프롬프트:**")
                                    st.markdown(f"```\n{data.get('prompt', '')}\n```")
                                    
                                    st.markdown("### 🔍 Evidence 정보")
                                    evidence_tokens = data.get("evidence_tokens", [])
                                    evidence_indices = data.get("evidence_indices", [])
                                    
                                    # Evidence 토큰과 인덱스를 테이블로 표시
                                    evidence_data = []
                                    for idx, token in zip(evidence_indices, evidence_tokens):
                                        evidence_data.append({
                                            "인덱스": idx,
                                            "토큰": token
                                        })
                                    if evidence_data:
                                        st.table(pd.DataFrame(evidence_data))
                                    else:
                                        st.warning("Evidence 정보가 없습니다.")
                                    
                                    # 메타데이터 표시
                                    st.markdown("### 📊 메타데이터")
                                    meta_data = {
                                        "도메인": data.get("domain", ""),
                                        "모델": data.get("model", ""),
                                        "타임스탬프": data.get("timestamp", ""),
                                        "인덱스": data.get("index", "")
                                    }
                                    st.json(meta_data)
                                    break
                    except Exception as e:
                        st.error(f"데이터셋 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
                    
                    st.markdown("---")
                    st.subheader("어텐션 실험")
                    if st.button("어텐션 추출 및 토큰화 보기"):
                        with st.spinner("모델에서 어텐션 추출 중..."):
                            attentions, tokens, token_ids = get_attention_from_session(selected_prompt)
                        if attentions is None:
                            st.error("해당 모델은 지원되지 않거나 로드에 실패했습니다.")
                        else:
                            st.markdown(f"**토큰화 결과:** {' | '.join(tokens)}")
                            evidence_set = set(evidence_tokens)
                            colored = []
                            for t in tokens:
                                if t in evidence_set:
                                    colored.append(f"<span style='background-color: #ffe066'>{t}</span>")
                                else:
                                    colored.append(t)
                            st.markdown("**evidence 토큰 강조:**<br>" + ' '.join(colored), unsafe_allow_html=True)
                            st.info(f"어텐션 shape: {len(attentions)} layers, {attentions[-1].shape[1]} heads, {attentions[-1].shape[2]} tokens")
                            last_attn = attentions[-1][0]
                            fig, avg_evidence_attention = plot_attention_head_heatmap(last_attn, tokens, evidence_indices)
                            st.pyplot(fig)
                            max_head = int(np.argmax(avg_evidence_attention))
                            st.success(f"가장 evidence에 강하게 반응하는 헤드: Head {max_head} (평균 어텐션 {avg_evidence_attention[max_head]:.4f})")
                            fig2, avg_attn = plot_attention_head_token_heatmap(last_attn, tokens, evidence_indices)
                            st.markdown("---")
                            st.subheader("헤드별 토큰 어텐션 히트맵")
                            st.pyplot(fig2)
                            st.caption("* x축: 토큰 인덱스 (evidence 토큰은 빨간색), y축: 헤드 인덱스, 값: 어텐션 평균 *")
                            st.markdown("---")
                            st.subheader("토큰별 헤드 어텐션 히트맵")
                            fig3, avg_attn_t = plot_token_head_attention_heatmap(last_attn, tokens, evidence_indices)
                            st.pyplot(fig3)
                            st.caption("* y축: 토큰 인덱스(문자열, evidence 토큰은 빨간색), x축: 헤드 인덱스, 값: 어텐션 평균 *")
                else:
                    st.warning("해당 파일에서 프롬프트를 불러올 수 없습니다.")
            else:
                st.warning(f"{selected_domain} 도메인에 데이터셋 파일이 없습니다.")
        else:
            st.warning(f"⚠️ {loaded_model_name} 모델용 데이터셋이 준비되지 않았습니다. 위의 '모델별 데이터셋 준비' 버튼을 클릭하여 데이터셋을 준비해주세요.")

        # 일괄 실험 섹션 추가
        st.markdown("---")
        st.subheader(":rocket: 모델별 일괄 실험")
        
        if has_model_dataset:
            # 실험 설정
            num_prompts = st.number_input("도메인별 프롬프트 수", min_value=1, max_value=10000, value=5)
            
            # 실험 시작 버튼
            if st.button("모델별 실험 시작"):
                st.info("실험을 시작합니다. 이 작업은 시간이 오래 걸릴 수 있습니다.")
                
                # 실험 실행
                try:
                    with st.spinner("실험을 실행하는 중입니다..."):
                        # 전체 도메인에 대해 실험 실행
                        all_results = []
                        total_domains = len([d for d, files in model_dataset_files.items() if files])
                        current_domain = 0
                        
                        for domain, file_list in model_dataset_files.items():
                            if not file_list:
                                continue
                            
                            current_domain += 1
                            st.text(f"진행 중: {domain} 도메인 ({current_domain}/{total_domains})")
                            
                            results = batch_domain_experiment(loaded_model_name, {domain: file_list}, num_prompts)
                            all_results.extend(results)
                        
                        # 실험 결과 저장
                        if all_results:
                            filename = save_experiment_result(all_results, loaded_model_name)
                            st.success(f"✅ 실험이 완료되었습니다!")
                            st.success(f"📁 결과가 저장되었습니다: {filename}")
                            st.info(f"📊 총 {len(all_results)}개의 실험 결과가 생성되었습니다.")
                            
                            # 결과 미리보기
                            st.markdown("### 📋 실험 결과 미리보기")
                            preview_data = []
                            for result in all_results[:5]:  # 처음 5개만 표시
                                preview_data.append({
                                    "도메인": result["domain"],
                                    "최대 어텐션 헤드": result["max_head"],
                                    "평균 어텐션": f"{result['avg_evidence_attention']:.4f}",
                                    "Evidence 토큰 수": len(result["evidence_indices"])
                                })
                            if preview_data:
                                st.table(pd.DataFrame(preview_data))
                        else:
                            st.warning("실험 결과가 생성되지 않았습니다.")
                            
                except Exception as e:
                    st.error(f"실험 중 오류가 발생했습니다: {str(e)}")
                    st.error("모델이 제대로 로드되어 있는지 확인해주세요.")
        else:
            st.warning("모델별 데이터셋이 준비되지 않아 일괄 실험을 수행할 수 없습니다.")
    else:
        st.warning("모델이 로드되어 있지 않습니다. 실험을 진행하려면 먼저 모델을 로드해주세요.")