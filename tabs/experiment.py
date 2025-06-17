import streamlit as st
import os
import json
from utils import get_available_models
import pandas as pd
from datetime import datetime

# Huggingface 관련 추가
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_ROOT = "dataset"

# 글로벌 모델 캐시
MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None
}

# 각 도메인별 데이터셋 파일 목록을 가져오는 함수
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

# 선택한 데이터셋 파일에서 프롬프트 리스트를 추출하는 함수 (최대 10000개)
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
    }
    hf_model = model_map.get(model_name)
    if not hf_model:
        st.error("지원하지 않는 모델명입니다.")
        return False
    try:
        # 모델과 토크나이저를 서버 메모리에 로드
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            quantization_config=quant_config,
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
        return True
    except Exception as e:
        st.error(f"모델 로딩 실패: {str(e)}")
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
    # 각 헤드별로 evidence 토큰(to_token)에 대한 어텐션 평균 계산
    # attn shape: (head, from_token, to_token)
    head_count = attn.shape[0]
    avg_evidence_attention = []
    for h in range(head_count):
        # from_token 전체에서 evidence 토큰(to_token)으로 가는 어텐션 평균
        avg = attn[h, :, evidence_indices].mean()
        avg_evidence_attention.append(avg)
    avg_evidence_attention = np.array(avg_evidence_attention)

    # 히트맵 시각화
    fig, ax = plt.subplots(figsize=(10, 1.5))
    sns.heatmap(avg_evidence_attention[None, :], annot=True, fmt=".2f", cmap="YlOrRd", cbar=True, ax=ax)
    ax.set_xlabel("Head index")
    ax.set_yticks([])
    ax.set_title("각 헤드별 evidence 토큰 어텐션 평균")
    plt.tight_layout()
    return fig, avg_evidence_attention

def plot_attention_head_token_heatmap(attn, tokens, evidence_indices):
    """
    attn: (head, from_token, to_token) numpy array (마지막 레이어)
    tokens: 토큰 리스트
    evidence_indices: evidence 토큰의 인덱스 리스트
    """
    # 각 헤드별로 to_token(모든 토큰)에 대한 어텐션 평균 (from_token 전체 평균)
    # shape: (head, to_token)
    avg_attn = attn.mean(axis=1)  # (head, to_token)

    fig, ax = plt.subplots(figsize=(min(1.2+0.4*len(tokens), 16), 1.5+0.3*avg_attn.shape[0]))
    sns.heatmap(avg_attn, annot=False, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_xlabel("토큰 인덱스")
    ax.set_ylabel("헤드 인덱스")
    ax.set_title("각 헤드별 토큰 어텐션 평균 히트맵")
    # x축에 토큰 라벨 표시 (길면 잘림)
    token_labels = [t if i not in evidence_indices else f"*{t}*" for i, t in enumerate(tokens)]
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
    # evidence 토큰 강조 (x축 라벨 색상)
    for idx in evidence_indices:
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
    # 각 토큰별로 헤드 어텐션 평균 (from_token 전체 평균)
    # shape: (head, to_token)
    avg_attn = attn.mean(axis=1)  # (head, to_token)
    # shape를 (to_token, head)로 transpose
    avg_attn_t = avg_attn.T  # (to_token, head)
    fig, ax = plt.subplots(figsize=(min(1.2+0.4*len(tokens), 16), 1.5+0.3*avg_attn_t.shape[1]))
    sns.heatmap(avg_attn_t, annot=False, cmap="YlGnBu", cbar=True, ax=ax)
    ax.set_ylabel("토큰 인덱스")
    ax.set_xlabel("헤드 인덱스")
    ax.set_title("토큰별 헤드 어텐션 평균 히트맵")
    # y축에 토큰 라벨 표시 (길면 잘림)
    token_labels = [t if i not in evidence_indices else f"*{t}*" for i, t in enumerate(tokens)]
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(token_labels, rotation=0, fontsize=8)
    # evidence 토큰 강조 (y축 라벨 색상)
    for idx in evidence_indices:
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
    else:
        # 모델명 변환
        model_map = {
            'mistral:7b': 'mistralai/Mistral-7B-v0.1',
            'llama2:7b': 'meta-llama/Llama-2-7b-hf',
        }
        hf_model = model_map.get(model_name)
        if not hf_model:
            return results
        try:
            # 서버 메모리에 새로 모델 로드
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        except Exception as e:
            return results

    for domain, file_list in files.items():
        if not file_list:
            continue
        selected_file = file_list[0]  # 각 도메인 첫 번째 파일 사용(확장 가능)
        prompts = get_prompts(domain, selected_file, max_count=10000)
        if not prompts:
            continue
        # 프롬프트 샘플링
        sampled = prompts[:num_prompts]
        path = os.path.join(DATASET_ROOT, domain, selected_file)
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
                    avg = last_attn[h, :, evidence_indices].mean() if evidence_indices else 0.0
                    avg_evidence_attention.append(avg)
                max_head = int(np.argmax(avg_evidence_attention))
                results.append({
                    "domain": domain,
                    "prompt": prompt,
                    "max_head": max_head,
                    "avg_evidence_attention": avg_evidence_attention[max_head],
                    "evidence_indices": evidence_indices,
                    "tokens": tokens
                })
            except Exception as e:
                continue

    # 세션에서 로드한 모델이 아닌 경우에만 서버 메모리에서 해제
    if 'model' not in st.session_state:
        del model
        torch.cuda.empty_cache()
    return results

def save_experiment_result(results, model_name):
    """실험 결과를 experiment_results 폴더에 저장 (numpy 타입을 모두 변환)"""
    os.makedirs("experiment_results", exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results/{now}_{model_name.replace(':','_')}.json"
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

def check_model_loaded():
    """
    서버 메모리에 모델이 로드되어 있는지 확인합니다.
    세션에 없으면 글로벌 캐시에서 복구합니다.
    """
    try:
        if 'model' in st.session_state and 'tokenizer' in st.session_state:
            model = st.session_state['model']
            if model is not None:
                return True, st.session_state.get('model_name', '알 수 없는 모델')
        # 세션에 없으면 글로벌에서 복구
        elif MODEL_CACHE["model"] is not None and MODEL_CACHE["tokenizer"] is not None:
            st.session_state['model'] = MODEL_CACHE["model"]
            st.session_state['tokenizer'] = MODEL_CACHE["tokenizer"]
            st.session_state['model_name'] = MODEL_CACHE["model_name"]
            return True, MODEL_CACHE["model_name"]
        return False, None
    except Exception as e:
        print(f"모델 체크 중 에러 발생: {str(e)}")
        return False, None

def show():
    st.title("실험 탭")
    st.write("실험 탭입니다. (어텐션 히트맵 및 evidence 분석 기능이 여기에 추가될 예정)")

    st.markdown("---")
    st.subheader(":rocket: 모델 로드/해제")
    
    # 모델 로드 상태 확인
    is_loaded, loaded_model_name = check_model_loaded()
    if is_loaded:
        st.success(f"현재 {loaded_model_name} 모델이 서버 메모리에 로드되어 있습니다.")
    
    model_list = get_available_models()
    if model_list:
        selected_model = st.selectbox("사용할 모델을 선택하세요", model_list, key="session_model")
    else:
        selected_model = st.text_input("사용할 모델명을 직접 입력하세요 (예: llama2:7b)", key="session_model_input")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("모델 로드"):
            if is_loaded:
                st.warning(f"이미 {loaded_model_name} 모델이 로드되어 있습니다. 다른 모델을 로드하려면 먼저 현재 모델을 해제해주세요.")
            else:
                if load_model_to_session(selected_model):
                    st.success(f"모델 {selected_model}이(가) 서버 메모리에 로드되었습니다.")
                    # 모델 로드 후 상태 업데이트
                    is_loaded, loaded_model_name = check_model_loaded()
    with col2:
        if st.button("모델 해제"):
            if not is_loaded:
                st.warning("로드된 모델이 없습니다.")
            else:
                unload_model_from_session()
                # 모델 해제 후 상태 업데이트
                is_loaded, loaded_model_name = check_model_loaded()

    # 모델이 로드되어 있으면 실험 UI
    if is_loaded:
        files = get_dataset_files()
        domains = list(files.keys())
        selected_domain = st.selectbox("도메인 선택", domains)
        dataset_files = files[selected_domain]
        if dataset_files:
            selected_file = st.selectbox("데이터셋 파일 선택", dataset_files)
            prompts = get_prompts(selected_domain, selected_file)
            if prompts:
                selected_prompt = st.selectbox("프롬프트 선택 (최대 100개 미리보기)", prompts)
                prompt_idx = prompts.index(selected_prompt)
                path = os.path.join(DATASET_ROOT, selected_domain, selected_file)
                evidence_tokens = []
                evidence_indices = []
                try:
                    with open(path, "r") as f:
                        for i, line in enumerate(f):
                            if i == prompt_idx:
                                data = json.loads(line)
                                evidence_tokens = data.get("evidence_tokens", [])
                                evidence_indices = data.get("evidence_indices", [])
                                break
                except Exception:
                    pass
                st.markdown("**evidence tokens:** " + ", ".join([str(t) for t in evidence_tokens]))
                st.markdown("**evidence indices:** " + ", ".join([str(idx) for idx in evidence_indices]))
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

        # 일괄 실험 섹션 추가
        st.markdown("---")
        st.subheader(":rocket: 일괄 실험")
        
        # 실험 설정
        num_prompts = st.number_input("도메인별 프롬프트 수", min_value=1, max_value=10000, value=5)
        
        # 실험 시작/중지 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("실험 시작"):
                st.session_state['experiment_running'] = True
                st.session_state['experiment_results'] = []
                
                # 진행 상황 표시를 위한 프로그레스 바
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 전체 도메인 수 계산
                total_domains = len([d for d, files in files.items() if files])
                current_domain = 0
                
                for domain, file_list in files.items():
                    if not file_list:
                        continue
                    
                    if not st.session_state.get('experiment_running', False):
                        st.warning("실험이 중단되었습니다.")
                        break
                    
                    current_domain += 1
                    status_text.text(f"진행 중: {domain} 도메인 ({current_domain}/{total_domains})")
                    progress_bar.progress(current_domain / total_domains)
                    
                    results = batch_domain_experiment(loaded_model_name, {domain: file_list}, num_prompts)
                    st.session_state['experiment_results'].extend(results)
                
                if st.session_state.get('experiment_running', False):
                    # 실험 결과 저장
                    filename = save_experiment_result(st.session_state['experiment_results'], loaded_model_name)
                    st.success(f"실험이 완료되었습니다. 결과가 {filename}에 저장되었습니다.")
                
                # 세션 상태 초기화
                st.session_state['experiment_running'] = False
                progress_bar.empty()
                status_text.empty()
        
        with col2:
            if st.button("실험 중단"):
                st.session_state['experiment_running'] = False
                st.warning("실험 중단 요청이 전송되었습니다. 현재 진행 중인 도메인이 완료되면 중단됩니다.")
    else:
        st.warning("모델이 로드되어 있지 않습니다. 실험을 진행하려면 먼저 모델을 로드해주세요.")