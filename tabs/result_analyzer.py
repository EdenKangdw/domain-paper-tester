import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from .experiment import get_model_experiment_path, list_model_experiment_results, load_model_experiment_result, analyze_head_attention_pattern

def show():
    st.title("🔍 실험 결과 분석")
    st.markdown("실험 결과를 로드하고 특정 헤드의 attention 패턴을 분석합니다.")
    
    # 모델 선택
    st.subheader("📊 실험 결과 선택")
    
    # 사용 가능한 모델들 찾기
    experiment_root = "experiment_results"
    if not os.path.exists(experiment_root):
        st.error("실험 결과 디렉토리가 없습니다. 먼저 실험을 실행해주세요.")
        return
    
    available_models = []
    for model_dir in os.listdir(experiment_root):
        model_path = os.path.join(experiment_root, model_dir)
        if os.path.isdir(model_path):
            # 디렉토리명을 모델명으로 변환
            model_name = model_dir.replace('_', '/').replace('_', ':')
            files = [f for f in os.listdir(model_path) if f.endswith(".json") and not f.endswith("_errors.json")]
            if files:
                available_models.append(model_name)
    
    if not available_models:
        st.error("분석할 실험 결과가 없습니다. 먼저 실험을 실행해주세요.")
        return
    
    selected_model = st.selectbox(
        "분석할 모델을 선택하세요",
        available_models,
        key="analyzer_model_selector"
    )
    
    # 실험 결과 파일 선택
    experiment_results = list_model_experiment_results(selected_model)
    
    if not experiment_results:
        st.error("선택한 모델의 실험 결과가 없습니다.")
        return
    
    selected_result = st.selectbox(
        "분석할 실험 결과를 선택하세요",
        experiment_results,
        key="analyzer_result_selector"
    )
    
    # 실험 결과 로드
    if st.button("📈 결과 로드", type="primary"):
        with st.spinner("실험 결과를 로드하는 중..."):
            result_data = load_model_experiment_result(selected_model, selected_result)
            
            if not result_data:
                st.error("실험 결과를 로드할 수 없습니다.")
                return
            
            st.session_state.analyzer_result_data = result_data
            st.session_state.analyzer_model_name = selected_model
            st.success(f"✅ {len(result_data)}개 결과 로드 완료!")
    
    # 결과 데이터가 있으면 분석 시작
    if 'analyzer_result_data' in st.session_state and st.session_state.analyzer_result_data:
        result_data = st.session_state.analyzer_result_data
        
        st.subheader("📊 결과 요약")
        
        # 도메인별 통계
        domain_stats = {}
        head_stats = {}
        
        for result in result_data:
            domain = result.get("domain", "unknown")
            max_head = result.get("max_head", -1)
            # 새로운 변수명이 있으면 사용, 없으면 기존 변수명 사용
            attention_score = result.get("avg_evidence_attention_whole", result.get("avg_evidence_attention", 0.0))
            
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "total_attention": 0.0, "heads": []}
            domain_stats[domain]["count"] += 1
            domain_stats[domain]["total_attention"] += attention_score
            domain_stats[domain]["heads"].append(max_head)
            
            if max_head not in head_stats:
                head_stats[max_head] = {"count": 0, "total_attention": 0.0, "domains": []}
            head_stats[max_head]["count"] += 1
            head_stats[max_head]["total_attention"] += attention_score
            head_stats[max_head]["domains"].append(domain)
        
        # 도메인별 통계 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**도메인별 통계**")
            for domain, stats in domain_stats.items():
                avg_attention = stats["total_attention"] / stats["count"]
                most_common_head = max(set(stats["heads"]), key=stats["heads"].count)
                st.write(f"**{domain}**: {stats['count']}개 결과, 평균 어텐션: {avg_attention:.4f}, 가장 많이 선택된 헤드: {most_common_head}")
        
        with col2:
            st.markdown("**헤드별 통계**")
            # 상위 10개 헤드만 표시
            sorted_heads = sorted(head_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
            for head, stats in sorted_heads:
                avg_attention = stats["total_attention"] / stats["count"]
                st.write(f"**Head {head}**: {stats['count']}회 선택, 평균 어텐션: {avg_attention:.4f}")
        
        # 특정 헤드 분석
        st.subheader("🔍 특정 헤드 분석")
        
        # 분석할 헤드 선택
        target_head = st.number_input(
            "분석할 헤드 번호",
            min_value=0,
            max_value=63,  # 일반적인 transformer 모델의 헤드 수
            value=27,
            help="분석하고 싶은 특정 헤드의 번호를 입력하세요"
        )
        
        if st.button("🔬 헤드 분석", type="primary"):
            # 해당 헤드가 선택된 결과들 찾기
            head_results = [r for r in result_data if r.get("max_head") == target_head]
            
            if not head_results:
                st.warning(f"Head {target_head}이 선택된 결과가 없습니다.")
                return
            
            st.success(f"Head {target_head}이 선택된 {len(head_results)}개 결과를 분석합니다.")
            
            # 첫 번째 결과로 attention 패턴 분석
            sample_result = head_results[0]
            
            # 모델이 로드되어 있는지 확인 (더 자세한 디버깅)
            st.info(f"세션 상태 확인: model={'model' in st.session_state}, tokenizer={'tokenizer' in st.session_state}")
            
            if 'model' in st.session_state and 'tokenizer' in st.session_state:
                model = st.session_state['model']
                tokenizer = st.session_state['tokenizer']
                
                st.info(f"모델 타입: {type(model)}, 토크나이저 타입: {type(tokenizer)}")
                
                if model is None or tokenizer is None:
                    st.error("모델 또는 토크나이저가 None입니다. 모델 로드 탭에서 모델을 다시 로드해주세요.")
                    return
                
                st.success("✅ 모델이 정상적으로 로드되어 있습니다.")
                
                # attention 추출
                prompt = sample_result.get("prompt", "")
                evidence_indices = sample_result.get("evidence_indices", [])
                
                if not prompt:
                    st.error("프롬프트 정보가 없습니다.")
                    return
                
                with st.spinner("Attention 패턴을 분석하는 중..."):
                    try:
                        import torch
                        inputs = tokenizer(prompt, return_tensors="pt")
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model(**inputs, output_attentions=True)
                            attentions = tuple(attn.cpu().numpy() for attn in outputs.attentions)
                        
                        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                        last_attn = attentions[-1][0]  # (head, from_token, to_token)
                        
                        # Head 27의 attention 패턴 분석
                        fig, stats, head_attention = analyze_head_attention_pattern(
                            last_attn, tokens, evidence_indices, target_head
                        )
                        
                        if fig is not None:
                            st.pyplot(fig)
                            
                            # 통계 정보 표시
                            st.subheader("📈 분석 결과")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Evidence 토큰 평균 어텐션", f"{stats['evidence_mean']:.4f}")
                            with col2:
                                st.metric("Non-Evidence 토큰 평균 어텐션", f"{stats['non_evidence_mean']:.4f}")
                            with col3:
                                st.metric("어텐션 비율", f"{stats['attention_ratio']:.2f}")
                            
                            # 해석
                            st.subheader("💡 해석")
                            if stats['attention_ratio'] > 1.5:
                                st.success(f"✅ Head {target_head}은 evidence 토큰에 강하게 반응합니다! (비율: {stats['attention_ratio']:.2f})")
                            elif stats['attention_ratio'] > 1.1:
                                st.info(f"ℹ️ Head {target_head}은 evidence 토큰에 약간 더 반응합니다. (비율: {stats['attention_ratio']:.2f})")
                            else:
                                st.warning(f"⚠️ Head {target_head}은 evidence 토큰에 특별히 반응하지 않습니다. (비율: {stats['attention_ratio']:.2f})")
                            
                            # 상세 통계
                            st.markdown("**상세 통계**")
                            st.json(stats)
                            
                        else:
                            st.error("Attention 패턴 분석에 실패했습니다.")
                            
                    except Exception as e:
                        st.error(f"Attention 분석 중 오류 발생: {str(e)}")
            else:
                st.error("모델이 로드되지 않았습니다. 모델 로드 탭에서 모델을 로드해주세요.")
                st.info("현재 세션 상태 키들: " + str(list(st.session_state.keys())))
        
        # 전체 헤드 분포 시각화
        st.subheader("📊 전체 헤드 분포")
        
        if st.button("📈 분포 시각화", type="secondary"):
            # 헤드별 선택 횟수
            head_counts = {head: stats["count"] for head, stats in head_stats.items()}
            
            fig, ax = plt.subplots(figsize=(12, 6))
            heads = list(head_counts.keys())
            counts = list(head_counts.values())
            
            bars = ax.bar(heads, counts, alpha=0.7, color='skyblue')
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Selection Count')
            ax.set_title('Head Selection Distribution')
            ax.set_xticks(heads)
            ax.set_xticklabels(heads, rotation=45)
            
            # 상위 5개 헤드 강조
            sorted_heads = sorted(head_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for head, count in sorted_heads:
                if head in heads:
                    idx = heads.index(head)
                    bars[idx].set_color('red')
                    ax.text(head, count + 0.1, f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 상위 헤드 정보
            st.markdown("**가장 많이 선택된 헤드들**")
            for i, (head, count) in enumerate(sorted_heads, 1):
                avg_attention = head_stats[head]["total_attention"] / count
                st.write(f"{i}. **Head {head}**: {count}회 선택, 평균 어텐션: {avg_attention:.4f}")
    
    else:
        st.info("실험 결과를 로드하여 분석을 시작하세요.") 