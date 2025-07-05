import streamlit as st
import os
import json
import requests
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
    domains = ["economy", "technical", "legal", "medical"]
    for domain in domains:
        os.makedirs(os.path.join(dataset_path, domain), exist_ok=True)
    
    return experiment_path, dataset_path

def get_model_dataset_files(model_name):
    """특정 모델의 데이터셋 파일 목록을 가져옵니다."""
    dataset_path = get_model_dataset_path(model_name)
    domains = ["economy", "technical", "legal", "medical"]
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
        domains = ["economy", "technical", "legal", "medical"]
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
    domains = ["economy", "technical", "legal", "medical"]
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
    import os
    
    # Qwen 모델의 경우에만 특별한 환경 변수 설정
    if 'qwen' in model_name.lower():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # scaled_dot_product_attention 비활성화 (Qwen 모델만)
        os.environ["PYTORCH_DISABLE_SCALED_DOT_PRODUCT_ATTENTION"] = "1"
    # 기본 모델명 추출 (파라미터 수 무시)
    base_model = model_name.split(":")[0]
    
    model_map = {
        'mistral': 'mistralai/Mistral-7B-v0.1',
        'llama2': 'meta-llama/Llama-2-7b-hf',
        'gemma': 'google/gemma-7b',
        'qwen': 'Qwen/Qwen-7B',
        'deepseek': 'deepseek-ai/deepseek-llm-7b-base',
        'deepseek-r1': 'deepseek-ai/deepseek-llm-7b-base',
        'deepseek-r1-distill-llama': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'yi': '01-ai/Yi-6B',
        'openchat': 'openchat/openchat-3.5-7b',
        'neural': 'microsoft/DialoGPT-medium',
        'phi': 'microsoft/phi-2',
        'stable': 'stabilityai/stablelm-base-alpha-7b',
    }
    
    # 1. 전체 모델명으로 먼저 시도
    hf_model = model_map.get(model_name)
    if not hf_model:
        # 2. 기본 모델명으로 시도 (파라미터 수 무시)
        hf_model = model_map.get(base_model)
    
    if not hf_model:
        st.error(f"지원하지 않는 모델명입니다: {model_name} (기본 모델명: {base_model})")
        st.info(f"지원되는 모델들: {', '.join(model_map.keys())}")
        return False
    try:
        # 모델과 토크나이저를 서버 메모리에 로드
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
        
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
        
        # Qwen 모델의 경우 attention 구현을 강제로 eager로 설정
        if 'qwen' in model_name.lower():
            import os
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            # Qwen 모델의 config를 수정하여 eager attention 사용
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
            config.attn_implementation = "eager"
            config.use_flash_attention_2 = False
            config.use_cache = True
        
        # 모델 로딩 시도
        try:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            if 'qwen' in model_name.lower():
                # Qwen 모델의 경우 config를 사용하여 로드
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model,
                    config=config,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation="eager"
                )
        except Exception as quant_error:
            # 8비트 양자화 실패 시 16비트로 시도
            st.warning("8비트 양자화 로딩 실패, 16비트로 시도합니다...")
            try:
                if 'qwen' in model_name.lower():
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_model,
                        config=config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_model,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation="eager"
                    )
            except Exception as device_error:
                # device_map 실패 시 CPU로 시도
                st.warning("자동 디바이스 매핑 실패, CPU로 로드합니다...")
                if 'qwen' in model_name.lower():
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_model,
                        config=config,
                        trust_remote_code=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        hf_model,
                        trust_remote_code=True,
                        attn_implementation="eager"
                    )
        
        # 8비트 양자화된 모델은 이미 올바른 디바이스로 설정되어 있음
        # device_map="auto"를 사용한 경우 추가 디바이스 이동 불필요
        try:
            # 모델의 현재 디바이스 확인
            device = next(model.parameters()).device
            st.info(f"모델이 {device} 디바이스에 로드되었습니다.")
        except Exception as device_error:
            st.warning(f"디바이스 확인 실패: {str(device_error)}")
            # 기본값으로 CPU 설정
            device = torch.device("cpu")
        
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
    # 모델의 디바이스 확인 및 입력 이동
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as device_error:
        # 디바이스 확인 실패 시 CPU 사용
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
    
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

def analyze_head_attention_pattern(attn, tokens, evidence_indices, target_head=27):
    """
    특정 헤드의 attention 패턴을 분석하여 evidence 토큰에만 특별히 반응하는지 확인
    attn: (head, from_token, to_token) numpy array (마지막 레이어)
    tokens: 토큰 리스트
    evidence_indices: evidence 토큰의 인덱스 리스트
    target_head: 분석할 헤드 번호
    """
    # evidence_indices 유효성 검사 및 필터링
    if not evidence_indices:
        evidence_indices = []
    else:
        # 토큰 길이를 벗어나는 인덱스 제거
        evidence_indices = [i for i in evidence_indices if 0 <= i < len(tokens)]
    
    if target_head >= attn.shape[0]:
        return None, None, None
    
    # 특정 헤드의 attention 패턴 (from_token 전체 평균)
    head_attention = attn[target_head].mean(axis=0)  # (to_token,)
    
    # evidence 토큰과 non-evidence 토큰 분리
    evidence_attention = head_attention[evidence_indices] if evidence_indices else np.array([])
    non_evidence_indices = [i for i in range(len(tokens)) if i not in evidence_indices]
    non_evidence_attention = head_attention[non_evidence_indices] if non_evidence_indices else np.array([])
    
    # 통계 계산
    stats = {
        'evidence_mean': float(evidence_attention.mean()) if len(evidence_attention) > 0 else 0.0,
        'evidence_std': float(evidence_attention.std()) if len(evidence_attention) > 0 else 0.0,
        'non_evidence_mean': float(non_evidence_attention.mean()) if len(non_evidence_attention) > 0 else 0.0,
        'non_evidence_std': float(non_evidence_attention.std()) if len(non_evidence_attention) > 0 else 0.0,
        'evidence_count': len(evidence_attention),
        'non_evidence_count': len(non_evidence_attention),
        'attention_ratio': float(evidence_attention.mean() / non_evidence_attention.mean()) if len(non_evidence_attention) > 0 and non_evidence_attention.mean() > 0 else 0.0
    }
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. 전체 attention 패턴
    ax1.bar(range(len(tokens)), head_attention, alpha=0.7, color='lightblue')
    # evidence 토큰 강조
    for idx in evidence_indices:
        ax1.bar(idx, head_attention[idx], color='red', alpha=0.8)
    ax1.set_xlabel('Token Index')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title(f'Head {target_head} Attention Pattern')
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels([t[:10] + '...' if len(t) > 10 else t for t in tokens], rotation=45, ha='right')
    
    # 2. Evidence vs Non-evidence 비교
    categories = ['Evidence Tokens', 'Non-Evidence Tokens']
    means = [stats['evidence_mean'], stats['non_evidence_mean']]
    stds = [stats['evidence_std'], stats['non_evidence_std']]
    
    bars = ax2.bar(categories, means, yerr=stds, capsize=5, alpha=0.7, color=['red', 'lightblue'])
    ax2.set_ylabel('Average Attention Weight')
    ax2.set_title(f'Head {target_head}: Evidence vs Non-Evidence Attention')
    
    # 값 표시
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001, 
                f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig, stats, head_attention

def batch_domain_experiment_multi_models(model_names, files, num_prompts=20):
    """
    여러 모델에 대해 evidence 어텐션 실험을 일괄 수행하고 통계 집계
    model_names: 사용할 모델명 리스트
    files: {domain: filename} 형태의 선택된 파일들
    num_prompts: 도메인별 샘플링할 프롬프트 개수
    """
    import time
    from datetime import datetime, timedelta
    
    all_results = []
    error_logs = []  # 에러 로그 저장용
    
    # 전체 작업량 계산 (모델 수 * 도메인 수 * 프롬프트 수)
    total_tasks = len(model_names) * len(files) * num_prompts
    completed_tasks = 0
    start_time = time.time()
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 모델별 진행 상황 추적
    model_progress = {}
    for model_name in model_names:
        model_progress[model_name] = {}
        for domain in files.keys():
            model_progress[model_name][domain] = 0
    
    for model_name in model_names:
        st.info(f"🔄 {model_name} 모델 실험 시작...")
        
        # 각 모델에 대해 실험 수행
        model_results = batch_domain_experiment_single_model(model_name, files, num_prompts, 
                                                           progress_bar, status_text, 
                                                           model_progress, completed_tasks, 
                                                           total_tasks, start_time)
        
        all_results.extend(model_results)
        
        # 모델별 결과 저장
        if model_results:
            save_experiment_result(model_results, model_name)
            st.success(f"✅ {model_name} 모델 실험 완료! {len(model_results)}개 결과")
        
        # 모델 언로드 (메모리 절약)
        unload_model_from_session()
    
    # 전체 결과 요약
    if all_results:
        st.success(f"🎉 모든 모델 실험 완료! 총 {len(all_results)}개 결과")
        
        # 모델별 결과 요약
        model_summary = {}
        for result in all_results:
            model_name = result['model_name']
            if model_name not in model_summary:
                model_summary[model_name] = {'count': 0, 'domains': set()}
            model_summary[model_name]['count'] += 1
            model_summary[model_name]['domains'].add(result['domain'])
        
        st.subheader("📊 실험 결과 요약")
        for model_name, summary in model_summary.items():
            domains_str = ', '.join(sorted(summary['domains']))
            st.info(f"**{model_name}**: {summary['count']}개 결과 ({domains_str} 도메인)")
    
    return all_results

def batch_domain_experiment_single_model(model_name, files, num_prompts=20, 
                                       progress_bar=None, status_text=None, 
                                       model_progress=None, completed_tasks=0, 
                                       total_tasks=0, start_time=None):
    """
    단일 모델에 대해 evidence 어텐션 실험을 수행
    """
    import time
    from datetime import datetime, timedelta
    
    results = []
    error_logs = []  # 에러 로그 저장용
    
    # 모델 로드
    with st.spinner(f"{model_name} 모델을 로드하는 중..."):
        success = load_model_to_session(model_name)
        if not success:
            st.error(f"❌ {model_name} 모델 로드에 실패했습니다.")
            return results
    
    # 서버 메모리에 로드된 모델이 있는지 확인
    if 'model' in st.session_state and 'tokenizer' in st.session_state:
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        
        # 모델과 토크나이저가 실제로 None이 아닌지 확인
        if model is None or tokenizer is None:
            st.error("모델 또는 토크나이저가 None입니다. 모델을 다시 로드해주세요.")
            return results
    else:
        st.error("모델이 세션에 로드되어 있지 않습니다. 실험을 진행하려면 먼저 모델을 로드해주세요.")
        return results

    # 도메인별 진행 상황 추적
    domain_progress = {}
    for domain in files.keys():
        domain_progress[domain] = 0
    
    for domain, selected_file in files.items():
        if not selected_file:
            continue
        
        # 모델별 데이터셋에서 프롬프트 가져오기 (최대 10000개)
        prompts = get_model_prompts(model_name, domain, selected_file, max_count=10000)
        if not prompts:
            continue
        
        # 프롬프트 샘플링
        sampled = prompts[:num_prompts]
        
        # 모델별 데이터셋 경로 사용
        path = os.path.join(get_model_dataset_path(model_name), domain, selected_file)
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            continue
            
        domain_results = 0
        for i, prompt in enumerate(sampled):
            try:
                idx = prompts.index(prompt)
                data = json.loads(lines[idx])
                evidence_indices = data.get("evidence_indices", [])
                
                # 어텐션 추출
                inputs = tokenizer(prompt, return_tensors="pt")
                # 모델의 디바이스 확인 및 입력 이동
                try:
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception as device_error:
                    # 디바이스 확인 실패 시 CPU 사용
                    inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                with torch.no_grad():
                    try:
                        outputs = model(**inputs, output_attentions=True)
                        attentions = tuple(attn.cpu().numpy() for attn in outputs.attentions)
                    except Exception as attn_error:
                        continue
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # evidence_indices 타입 확인 및 안전한 처리
                if isinstance(evidence_indices, list):
                    evidence_indices = [i for i in evidence_indices if isinstance(i, (int, float)) and i < len(tokens)]
                else:
                    # evidence_indices가 리스트가 아닌 경우 빈 리스트로 초기화
                    evidence_indices = []
                
                last_attn = attentions[-1][0]  # (head, from_token, to_token)
                # 헤드별 evidence 어텐션 평균
                head_count = last_attn.shape[0]
                avg_evidence_attention_whole = []
                for h in range(head_count):
                    if evidence_indices:  # evidence_indices가 비어있지 않을 때만 계산
                        try:
                            avg = last_attn[h, :, evidence_indices].mean()
                        except (IndexError, ValueError):
                            avg = 0.0  # 인덱스 에러 발생 시 0 반환
                    else:
                        avg = 0.0  # evidence_indices가 비어있으면 0 반환
                    avg_evidence_attention_whole.append(avg)
                max_head = int(np.argmax(avg_evidence_attention_whole))
                results.append({
                    "domain": domain,
                    "prompt": prompt,
                    "max_head": max_head,
                    "avg_evidence_attention": avg_evidence_attention_whole[max_head],  # 기존 max값 (호환성 유지)
                    "avg_evidence_attention_whole": avg_evidence_attention_whole,  # 32차원 리스트 전체 저장
                    "evidence_indices": evidence_indices,
                    "tokens": tokens,
                    "model_name": model_name,
                    "tokenizer_name": tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "unknown"
                })
                domain_results += 1
                domain_progress[domain] = domain_results
                if model_progress and model_name in model_progress:
                    model_progress[model_name][domain] = domain_results
                completed_tasks += 1
                
                # 진행 상황 업데이트
                if progress_bar and status_text and start_time:
                    elapsed_time = time.time() - start_time
                    if completed_tasks > 0:
                        avg_time_per_task = elapsed_time / completed_tasks
                        remaining_tasks = total_tasks - completed_tasks
                        estimated_remaining_time = remaining_tasks * avg_time_per_task
                        
                        # 시간 포맷팅
                        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                        remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
                        
                        # 프로그레스바와 상태 정보 업데이트
                        progress_bar.progress(completed_tasks / total_tasks)
                        
                        # 모든 모델과 도메인의 진행 상황 표시
                        progress_info = []
                        for m, domains in model_progress.items():
                            for d, progress in domains.items():
                                progress_info.append(f"{m}-{d}: {progress}/{num_prompts}")
                        
                        status_text.write(f"**소요시간: {elapsed_str} / 남은 시간: {remaining_str}**  \n**{' | '.join(progress_info)}**")
                
            except Exception as e:
                completed_tasks += 1
                # 에러 로그 저장
                error_log = {
                    "model_name": model_name,
                    "domain": domain,
                    "prompt_index": i+1,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                error_logs.append(error_log)
                
                # 에러 로그를 상태 텍스트에 추가
                if progress_bar and status_text and start_time:
                    elapsed_time = time.time() - start_time
                    if completed_tasks > 0:
                        avg_time_per_task = elapsed_time / completed_tasks
                        remaining_tasks = total_tasks - completed_tasks
                        estimated_remaining_time = remaining_tasks * avg_time_per_task
                        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                        remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
                    else:
                        elapsed_str = "0:00:00"
                        remaining_str = "0:00:00"
                    
                    error_msg = f"❌ {model_name}-{domain} 도메인 {i+1}번째 프롬프트 처리 실패: {str(e)}"
                    
                    # 모든 모델과 도메인의 진행 상황 표시
                    progress_info = []
                    for m, domains in model_progress.items():
                        for d, progress in domains.items():
                            progress_info.append(f"{m}-{d}: {progress}/{num_prompts}")
                    
                    status_text.write(f"**소요시간: {elapsed_str} / 남은 시간: {remaining_str}**  \n**{' | '.join(progress_info)}**  \n{error_msg}")
                continue

    # 에러 로그가 있으면 파일에 저장
    if error_logs:
        experiment_path = get_model_experiment_path(model_name)
        os.makedirs(experiment_path, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = f"{experiment_path}/{now}_{model_name.replace(':','_')}_errors.json"
        with open(error_log_file, "w", encoding="utf-8") as f:
            json.dump(error_logs, f, ensure_ascii=False, indent=2)
        
        # 에러 개수 표시
        if status_text:
            status_text.write(f"✅ {model_name} 실험 완료! 총 {len(results)}개 결과, {len(error_logs)}개 에러 발생  \n에러 로그: {error_log_file}")
    
    return results

def batch_domain_experiment(model_name, files, num_prompts=20):
    """
    여러 도메인에 대해 evidence 어텐션 실험을 일괄 수행하고 통계 집계 (단일 모델용)
    model_name: 사용할 모델명
    files: {domain: filename} 형태의 선택된 파일들
    num_prompts: 도메인별 샘플링할 프롬프트 개수
    """
    import time
    from datetime import datetime, timedelta
    
    results = []
    error_logs = []  # 에러 로그 저장용
    
    # 서버 메모리에 로드된 모델이 있는지 확인
    if 'model' in st.session_state and 'tokenizer' in st.session_state:
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        
        # 모델과 토크나이저가 실제로 None이 아닌지 확인
        if model is None or tokenizer is None:
            st.error("모델 또는 토크나이저가 None입니다. 모델을 다시 로드해주세요.")
            return results
    else:
        st.error("모델이 세션에 로드되어 있지 않습니다. 실험을 진행하려면 먼저 모델을 로드해주세요.")
        return results

    # 전체 작업량 계산
    total_tasks = len(files) * num_prompts
    completed_tasks = 0
    start_time = time.time()
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 도메인별 진행 상황 추적
    domain_progress = {}
    for domain in files.keys():
        domain_progress[domain] = 0
    
    for domain, selected_file in files.items():
        if not selected_file:
            continue
        
        # 모델별 데이터셋에서 프롬프트 가져오기 (최대 10000개)
        prompts = get_model_prompts(model_name, domain, selected_file, max_count=10000)
        if not prompts:
            continue
        
        # 프롬프트 샘플링
        sampled = prompts[:num_prompts]
        
        # 모델별 데이터셋 경로 사용
        path = os.path.join(get_model_dataset_path(model_name), domain, selected_file)
        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            continue
            
        domain_results = 0
        for i, prompt in enumerate(sampled):
            try:
                idx = prompts.index(prompt)
                data = json.loads(lines[idx])
                evidence_indices = data.get("evidence_indices", [])
                
                # 어텐션 추출
                inputs = tokenizer(prompt, return_tensors="pt")
                # 모델의 디바이스 확인 및 입력 이동
                try:
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception as device_error:
                    # 디바이스 확인 실패 시 CPU 사용
                    inputs = {k: v.to('cpu') for k, v in inputs.items()}
                
                with torch.no_grad():
                    try:
                        outputs = model(**inputs, output_attentions=True)
                        attentions = tuple(attn.cpu().numpy() for attn in outputs.attentions)
                    except Exception as attn_error:
                        continue
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # evidence_indices 타입 확인 및 안전한 처리
                if isinstance(evidence_indices, list):
                    evidence_indices = [i for i in evidence_indices if isinstance(i, (int, float)) and i < len(tokens)]
                else:
                    # evidence_indices가 리스트가 아닌 경우 빈 리스트로 초기화
                    evidence_indices = []
                
                last_attn = attentions[-1][0]  # (head, from_token, to_token)
                # 헤드별 evidence 어텐션 평균
                head_count = last_attn.shape[0]
                avg_evidence_attention_whole = []
                for h in range(head_count):
                    if evidence_indices:  # evidence_indices가 비어있지 않을 때만 계산
                        try:
                            avg = last_attn[h, :, evidence_indices].mean()
                        except (IndexError, ValueError):
                            avg = 0.0  # 인덱스 에러 발생 시 0 반환
                    else:
                        avg = 0.0  # evidence_indices가 비어있으면 0 반환
                    avg_evidence_attention_whole.append(avg)
                max_head = int(np.argmax(avg_evidence_attention_whole))
                results.append({
                    "domain": domain,
                    "prompt": prompt,
                    "max_head": max_head,
                    "avg_evidence_attention": avg_evidence_attention_whole[max_head],  # 기존 max값 (호환성 유지)
                    "avg_evidence_attention_whole": avg_evidence_attention_whole,  # 32차원 리스트 전체 저장
                    "evidence_indices": evidence_indices,
                    "tokens": tokens,
                    "model_name": model_name,
                    "tokenizer_name": tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "unknown"
                })
                domain_results += 1
                domain_progress[domain] = domain_results
                completed_tasks += 1
                
                # 진행 상황 업데이트
                elapsed_time = time.time() - start_time
                if completed_tasks > 0:
                    avg_time_per_task = elapsed_time / completed_tasks
                    remaining_tasks = total_tasks - completed_tasks
                    estimated_remaining_time = remaining_tasks * avg_time_per_task
                    
                    # 시간 포맷팅
                    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                    remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
                    
                    # 프로그레스바와 상태 정보 업데이트
                    progress_bar.progress(completed_tasks / total_tasks)
                    
                    # 모든 도메인의 진행 상황 표시
                    progress_info = []
                    for d, progress in domain_progress.items():
                        progress_info.append(f"{d}: {progress}/{num_prompts}")
                    
                    status_text.write(f"**소요시간: {elapsed_str} / 남은 시간: {remaining_str}**  \n**{' | '.join(progress_info)}**")
                
            except Exception as e:
                completed_tasks += 1
                # 에러 로그 저장
                error_log = {
                    "domain": domain,
                    "prompt_index": i+1,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                error_logs.append(error_log)
                
                # 에러 로그를 상태 텍스트에 추가
                elapsed_time = time.time() - start_time
                if completed_tasks > 0:
                    avg_time_per_task = elapsed_time / completed_tasks
                    remaining_tasks = total_tasks - completed_tasks
                    estimated_remaining_time = remaining_tasks * avg_time_per_task
                    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
                    remaining_str = str(timedelta(seconds=int(estimated_remaining_time)))
                else:
                    elapsed_str = "0:00:00"
                    remaining_str = "0:00:00"
                
                error_msg = f"❌ {domain} 도메인 {i+1}번째 프롬프트 처리 실패: {str(e)}"
                
                # 모든 도메인의 진행 상황 표시
                progress_info = []
                for d, progress in domain_progress.items():
                    progress_info.append(f"{d}: {progress}/{num_prompts}")
                
                status_text.write(f"**소요시간: {elapsed_str} / 남은 시간: {remaining_str}**  \n**{' | '.join(progress_info)}**  \n{error_msg}")
                continue

    # 에러 로그가 있으면 파일에 저장
    if error_logs:
        experiment_path = get_model_experiment_path(model_name)
        os.makedirs(experiment_path, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = f"{experiment_path}/{now}_{model_name.replace(':','_')}_errors.json"
        with open(error_log_file, "w", encoding="utf-8") as f:
            json.dump(error_logs, f, ensure_ascii=False, indent=2)
        
        # 에러 개수 표시
        status_text.write(f"✅ 실험 완료! 총 {len(results)}개 결과, {len(error_logs)}개 에러 발생  \n에러 로그: {error_log_file}")
    
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
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_model_name' not in st.session_state:
        st.session_state.current_model_name = None
    if 'dataset_files_cache' not in st.session_state:
        st.session_state.dataset_files_cache = {}
    if 'model_dataset_files_cache' not in st.session_state:
        st.session_state.model_dataset_files_cache = {}
    
    st.title("🔬 Attention Pattern Experiment")
    
    # 모델 선택 섹션
    st.subheader("🤖 Model Selection")
    
    # 모델 목록 (캐시됨)
    available_models = get_available_models()
    
    if not available_models:
        st.error("사용 가능한 모델이 없습니다. Model Load 탭에서 모델을 설치해주세요.")
        return
    
    # 실험 모드 선택
    experiment_mode = st.radio(
        "실험 모드를 선택하세요",
        ["단일 모델 실험", "다중 모델 실험"],
        key="experiment_mode_selector"
    )
    
    if experiment_mode == "단일 모델 실험":
        # 단일 모델 선택
        selected_model = st.selectbox(
            "실험할 모델을 선택하세요",
            available_models,
            key="experiment_model_selector"
        )
        
        # 모델 로드/언로드 버튼
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 모델 로드", type="primary", key="load_model_btn"):
                if selected_model != st.session_state.current_model_name:
                    # 다른 모델을 로드하는 경우
                    if st.session_state.model_loaded:
                        unload_model_from_session()
                    
                    with st.spinner(f"{selected_model} 모델을 로드하는 중..."):
                        success = load_model_to_session(selected_model)
                        if success:
                            st.session_state.model_loaded = True
                            st.session_state.current_model_name = selected_model
                            st.success(f"✅ {selected_model} 모델이 로드되었습니다!")
                        else:
                            st.error(f"❌ {selected_model} 모델 로드에 실패했습니다.")
                else:
                    st.info("이미 로드된 모델입니다.")
        
        with col2:
            if st.button("📤 모델 언로드", type="secondary", key="unload_model_btn"):
                if st.session_state.model_loaded:
                    unload_model_from_session()
                    st.session_state.model_loaded = False
                    st.session_state.current_model_name = None
                    st.success("✅ 모델이 언로드되었습니다!")
                else:
                    st.info("로드된 모델이 없습니다.")
        
        with col3:
            if st.button("🔄 모델 목록 새로고침", type="secondary", key="refresh_models_btn"):
                get_available_models.clear()
                st.success("모델 목록이 새로고침되었습니다!")
        
        # 현재 모델 상태 표시
        if st.session_state.model_loaded:
            st.success(f"✅ 현재 로드된 모델: {st.session_state.current_model_name}")
        else:
            st.warning("⚠️ 모델이 로드되지 않았습니다.")
        
        selected_models = [selected_model]
        
    else:
        # 다중 모델 선택
        st.markdown("**🔧 실험할 모델들을 선택하세요 (여러 개 선택 가능)**")
        selected_models = st.multiselect(
            "사용 가능한 모델들",
            available_models,
            default=[available_models[0]] if available_models else [],
            help="여러 모델을 선택하면 순차적으로 처리됩니다."
        )
        
        if selected_models:
            st.info(f"선택된 모델: {', '.join(selected_models)}")
            
            # 모델별 상태 확인
            st.markdown("**📊 모델 상태 확인**")
            for model in selected_models:
                # 간단한 상태 확인 (실제 로드하지 않고)
                try:
                    response = requests.post(
                        f"http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": "test",
                            "stream": False
                        },
                        timeout=5
                    )
                    if response.status_code == 200:
                        st.success(f"✅ {model} (사용 가능)")
                    else:
                        st.warning(f"⚠️ {model} (응답 오류)")
                except:
                    st.warning(f"⚠️ {model} (연결 실패)")
        else:
            st.warning("⚠️ 최소 하나의 모델을 선택해주세요.")
            selected_models = []
        
        # 모델 목록 새로고침 버튼
        if st.button("🔄 모델 목록 새로고침", type="secondary", key="refresh_models_multi_btn"):
            get_available_models.clear()
            st.success("모델 목록이 새로고침되었습니다!")
            st.rerun()
    
    # 데이터셋 선택 섹션
    st.subheader("📊 Dataset Selection")
    
    # 데이터셋 파일 목록 (캐시된 경우 사용)
    cache_key = f"dataset_files_{selected_model}"
    if cache_key not in st.session_state.dataset_files_cache:
        st.session_state.dataset_files_cache[cache_key] = get_model_dataset_files(selected_model)
    
    dataset_files = st.session_state.dataset_files_cache[cache_key]
    
    # 도메인별 데이터셋 파일 선택
    domains = ["economy", "technical", "legal", "medical"]
    selected_files = {}
    
    for domain in domains:
        files = dataset_files.get(domain, [])
        if files:
            selected_file = st.selectbox(
                f"{domain.capitalize()} 도메인 데이터셋",
                files,
                key=f"file_selector_{domain}"
            )
            selected_files[domain] = selected_file
        else:
            st.warning(f"{domain.capitalize()} 도메인에 데이터셋 파일이 없습니다.")
    
    # 실험 설정
    st.subheader("⚙️ Experiment Settings")
    
    col4, col5 = st.columns(2)
    
    with col4:
        num_prompts = st.number_input(
            "실험할 프롬프트 수",
            min_value=1,
            max_value=10000,
            value=20,
            help="각 도메인별로 실험할 프롬프트의 개수 (최대 10000개, 권장: 20-50개)"
        )
    
    with col5:
        batch_mode = st.checkbox(
            "배치 모드",
            value=True,
            help="모든 도메인에 대해 한 번에 실험을 실행합니다."
        )
    
    # 실험 실행 버튼
    if st.button("🚀 실험 시작", type="primary", key="start_experiment_btn"):
        if not selected_models:
            st.error("모델을 먼저 선택해주세요.")
            return
        
        if not selected_files:
            st.error("실험할 데이터셋 파일을 선택해주세요.")
            return
        
        # 실험 실행
        try:
            if experiment_mode == "단일 모델 실험":
                # 단일 모델 실험
                if not st.session_state.model_loaded:
                    st.error("모델을 먼저 로드해주세요.")
                    return
                
                if batch_mode:
                    results = batch_domain_experiment(selected_models[0], selected_files, num_prompts)
                    if results:
                        save_experiment_result(results, selected_models[0])
                        st.success(f"✅ 실험 완료! 총 {len(results)}개 결과")
                    else:
                        st.warning("⚠️ 실험 결과가 없습니다.")
                else:
                    st.info("단일 도메인 실험 모드는 준비 중입니다.")
            else:
                # 다중 모델 실험
                st.info(f"🔄 {len(selected_models)}개 모델에 대해 실험을 시작합니다...")
                results = batch_domain_experiment_multi_models(selected_models, selected_files, num_prompts)
                if results:
                    st.success(f"✅ 모든 모델 실험 완료! 총 {len(results)}개 결과")
                else:
                    st.warning("⚠️ 실험 결과가 없습니다.")
        except Exception as e:
            st.error(f"❌ 실험 실행 중 오류 발생: {str(e)}")
    
    # 실험 결과 확인
    st.subheader("📋 Experiment Results")
    
    if experiment_mode == "단일 모델 실험":
        # 단일 모델 실험 결과
        experiment_results = list_model_experiment_results(selected_models[0])
        
        if experiment_results:
            selected_result = st.selectbox(
                "확인할 실험 결과를 선택하세요",
                experiment_results,
                key="result_selector"
            )
            
            if selected_result:
                result_data = load_model_experiment_result(selected_models[0], selected_result)
                if result_data:
                    st.json(result_data)
        else:
            st.info("아직 실험 결과가 없습니다.")
    else:
        # 다중 모델 실험 결과
        st.markdown("**📊 모든 모델의 실험 결과**")
        
        for model_name in selected_models:
            with st.expander(f"📋 {model_name} 실험 결과", expanded=False):
                experiment_results = list_model_experiment_results(model_name)
                
                if experiment_results:
                    selected_result = st.selectbox(
                        f"{model_name} 결과 선택",
                        experiment_results,
                        key=f"result_selector_{model_name}"
                    )
                    
                    if selected_result:
                        result_data = load_model_experiment_result(model_name, selected_result)
                        if result_data:
                            st.json(result_data)
                else:
                    st.info(f"{model_name}의 실험 결과가 없습니다.")
        
        # 전체 실험 결과 요약
        st.markdown("**📈 전체 실험 결과 요약**")
        all_experiments = get_all_model_experiments()
        if all_experiments:
            # 최근 실험 결과들 표시
            recent_experiments = []
            for model_name, experiments in all_experiments.items():
                if experiments:
                    recent_experiments.append({
                        'model': model_name,
                        'latest': experiments[0],  # 가장 최근 결과
                        'count': len(experiments)
                    })
            
            # 최근 실험 결과들을 테이블로 표시
            if recent_experiments:
                st.markdown("**최근 실험 결과**")
                for exp in recent_experiments:
                    st.info(f"**{exp['model']}**: {exp['count']}개 결과 (최근: {exp['latest']})")
        else:
            st.info("아직 실험 결과가 없습니다.")
    
    # 캐시 초기화 버튼 (디버깅용)
    if st.sidebar.button("🗑️ 캐시 초기화", help="모든 캐시를 초기화합니다", key="clear_cache_experiment"):
        # 캐시 함수들 초기화
        get_available_models.clear()
        # 세션 상태 캐시 초기화
        keys_to_remove = ['model_loaded', 'current_model_name', 'dataset_files_cache', 
                         'model_dataset_files_cache', 'available_models']
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("캐시가 초기화되었습니다!")
        # st.rerun() 제거 - 페이지 새로고침 방지