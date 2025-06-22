import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def list_experiment_results():
    if not os.path.exists("experiment_results"):
        return []
    files = [f for f in os.listdir("experiment_results") if f.endswith(".json")]
    files.sort(reverse=True)
    return files

def load_experiment_result(filename):
    path = os.path.join("experiment_results", filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def show():
    st.title(":open_file_folder: 실험 기록")
    
    # 실험 개요를 expander로 변경
    with st.expander("📝 실험 개요", expanded=False):
        st.markdown("""
        이 실험은 다양한 도메인(General, Technical, Legal, Medical)의 데이터셋을 활용하여, 각 프롬프트에서 evidence 토큰에 대해 언어모델의 어텐션 헤드가 어떻게 반응하는지 분석합니다.
        
        **evidence의 의미:**
        - 데이터셋에서 `evidence`란, 각 프롬프트(질문)에 대해 정답을 도출하는 데 핵심적인 역할을 하는 단어나 구(토큰)를 의미합니다.
        - 예를 들어, 의학 도메인에서 "What are the main side effects of this medication?"라는 질문이 있다면, 정답이 되는 부작용 관련 단어들이 evidence로 지정됩니다.
        - 각 데이터셋 샘플에는 evidence 토큰의 인덱스(`evidence_indices`)와 실제 토큰(`evidence_tokens`) 정보가 포함되어 있습니다.
        
        **실험 목적:**
        - 언어모델의 각 어텐션 헤드가 evidence 토큰에 얼마나 집중하는지(어텐션을 주는지) 분석합니다.
        - 도메인별로 특정 헤드가 evidence에 더 민감하게 반응하는지, 또는 범용적으로 반응하는 헤드가 있는지 확인합니다.
        - 이를 통해 모델 내부의 해석 가능성(interpretability)과 도메인 특화/범용 어텐션 패턴을 탐구할 수 있습니다.
        
        **⚠️ 중요한 주의사항 - 토큰화 차이:**
        - **모델마다 토큰화가 다릅니다**: 같은 텍스트라도 모델마다 다른 토크나이저를 사용하므로 토큰화 결과가 다릅니다.
        - **Evidence 인덱스가 달라집니다**: 토큰화가 다르므로 evidence 토큰의 위치(인덱스)도 달라집니다.
        - **헤드 비교 시 주의**: 다른 모델 간의 헤드 비교는 토큰화 차이를 고려해야 합니다.
        - **같은 모델 내에서만 직접 비교**: 같은 모델의 실험 결과끼리만 직접적인 헤드 비교가 의미가 있습니다.
        """)
    
    result_files = list_experiment_results()
    if result_files:
        # 탭으로 구분
        tab1, tab2, tab3 = st.tabs(["📊 단일 실험 분석", "🔍 상세 해석", "📈 n차 실험 비교"])
        
        with tab1:
            selected_result = st.selectbox("조회할 실험 결과 파일을 선택하세요", result_files, key="single_analysis")
            
            if st.button("기본 분석 실행", key="basic_analysis"):
                loaded_results = load_experiment_result(selected_result)
                df = pd.DataFrame(loaded_results)
                
                # 모델 정보 표시
                if 'model_name' in df.columns and len(df) > 0:
                    model_name = df['model_name'].iloc[0] if df['model_name'].iloc[0] else "알 수 없음"
                    tokenizer_name = df['tokenizer_name'].iloc[0] if 'tokenizer_name' in df.columns and df['tokenizer_name'].iloc[0] else "알 수 없음"
                    
                    st.markdown(f"**🔧 사용된 모델:** {model_name}")
                    st.markdown(f"**🔤 토크나이저:** {tokenizer_name}")
                
                # 기본 데이터프레임 표시
                st.markdown("### 📊 기본 실험 결과")
                st.dataframe(df[["domain", "max_head", "avg_evidence_attention"]])
                
                # 기본 정보
                with st.expander("📊 기본 실험 정보", expanded=True):
                    st.markdown(f"#### 총 {len(df)}개 샘플 분석")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 샘플 수", len(df))
                    with col2:
                        st.metric("분석 도메인 수", len(df['domain'].unique()))
                    with col3:
                        st.metric("평균 Evidence Attention", f"{df['avg_evidence_attention'].mean():.4f}")
                    with col4:
                        st.metric("가장 많이 사용된 헤드", f"헤드 {df['max_head'].value_counts().idxmax()}")
                
                # 도메인별 헤드 분포 설명
                st.markdown("""
                ### 📈 도메인별 Evidence 반응 헤드 분포
                이 그래프는 각 도메인에서 evidence 토큰에 가장 강하게 반응한 어텐션 헤드의 분포를 보여줍니다.
                
                **해석 방법:**
                - x축: 어텐션 헤드 번호 (0부터 시작)
                - y축: 각 헤드가 evidence에 가장 강하게 반응한 횟수
                - 각 막대의 높이: 해당 헤드가 evidence에 가장 강하게 반응한 횟수를 나타냄
                
                **주요 인사이트:**
                - 특정 도메인에서 특정 헤드가 자주 선택된다면, 그 헤드가 해당 도메인의 evidence를 처리하는 데 특화되어 있을 수 있습니다.
                - 여러 도메인에서 동일한 헤드가 자주 선택된다면, 그 헤드가 일반적인 evidence 처리에 중요한 역할을 할 수 있습니다.
                """)
                hist = df.groupby(["domain", "max_head"]).size().unstack(fill_value=0)
                st.bar_chart(hist)
                st.caption("*도메인별로 evidence에 가장 강하게 반응한 헤드의 빈도 분포*")
                
                # 도메인-헤드 히트맵 설명
                st.markdown("""
                ### 🔥 도메인-헤드 Evidence 반응 히트맵
                이 히트맵은 각 도메인과 어텐션 헤드의 조합에서 evidence 토큰에 대한 반응 빈도를 시각화합니다.
                
                **해석 방법:**
                - x축: 어텐션 헤드 번호
                - y축: 도메인 (Medical, Legal, Technical, General)
                - 색상 강도: 해당 도메인-헤드 조합에서 evidence에 가장 강하게 반응한 횟수
                    - 밝은 색상(노란색/빨간색): 높은 빈도
                    - 어두운 색상: 낮은 빈도
                - 숫자: 실제 빈도 수
                
                **주요 인사이트:**
                1. **도메인 특화 헤드:**
                   - 특정 도메인에서만 밝은 색상을 보이는 헤드는 해당 도메인에 특화된 evidence 처리에 관여할 수 있습니다.
                   - 예: Medical 도메인에서만 높은 값을 보이는 헤드는 의학 관련 evidence 처리에 특화되었을 수 있습니다.
                
                2. **범용 헤드:**
                   - 여러 도메인에서 높은 값을 보이는 헤드는 일반적인 evidence 처리에 관여할 수 있습니다.
                   - 이러한 헤드들은 다양한 도메인의 evidence를 처리하는 데 중요한 역할을 할 수 있습니다.
                
                3. **도메인별 패턴:**
                   - 각 도메인별로 특정 헤드들에 집중되는 패턴이 있다면, 해당 도메인의 evidence 처리에 특화된 헤드 그룹이 있을 수 있습니다.
                   - 이러한 패턴은 모델이 각 도메인의 evidence를 어떻게 처리하는지 이해하는 데 도움이 됩니다.
                """)
                fig, ax = plt.subplots(figsize=(min(12, 2+0.5*hist.shape[1]), 1.5+0.5*hist.shape[0]))
                sns.heatmap(hist, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
                ax.set_xlabel("Head Index")
                ax.set_ylabel("Domain")
                ax.set_title("Domain-Head Evidence Response Frequency Heatmap")
                st.pyplot(fig)
                st.caption("*x축: 헤드 인덱스, y축: 도메인, 값: evidence에 가장 많이 반응한 횟수*")
        
        with tab2:
            selected_result = st.selectbox("조회할 실험 결과 파일을 선택하세요", result_files, key="detailed_analysis")
            
            if st.button("상세 해석 실행", key="detailed_analysis_btn"):
                loaded_results = load_experiment_result(selected_result)
                df = pd.DataFrame(loaded_results)
                
                # 기본 정보
                with st.expander("📊 기본 실험 정보", expanded=True):
                    st.markdown(f"#### 총 {len(df)}개 샘플 분석")
                    
                    # 모델 정보 표시
                    if 'model_name' in df.columns and len(df) > 0:
                        model_name = df['model_name'].iloc[0] if df['model_name'].iloc[0] else "알 수 없음"
                        tokenizer_name = df['tokenizer_name'].iloc[0] if 'tokenizer_name' in df.columns and df['tokenizer_name'].iloc[0] else "알 수 없음"
                        
                        st.markdown(f"**🔧 사용된 모델:** {model_name}")
                        st.markdown(f"**🔤 토크나이저:** {tokenizer_name}")
                        
                        if tokenizer_name != "알 수 없음":
                            st.info("""
                            **토큰화 정보:**
                            - 이 실험은 위의 토크나이저를 사용하여 텍스트를 토큰화했습니다.
                            - 다른 모델과 비교할 때는 토큰화 방식이 다를 수 있음을 고려해주세요.
                            """)
                    
                    st.dataframe(df[["domain", "max_head", "avg_evidence_attention"]])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 샘플 수", len(df))
                    with col2:
                        st.metric("분석 도메인 수", len(df['domain'].unique()))
                    with col3:
                        st.metric("평균 Evidence Attention", f"{df['avg_evidence_attention'].mean():.4f}")
                    with col4:
                        st.metric("가장 많이 사용된 헤드", f"헤드 {df['max_head'].value_counts().idxmax()}")

                # 도메인별 성능 분석
                with st.expander("📈 도메인별 성능 분석", expanded=False):
                    domain_stats = df.groupby('domain').agg({
                        'avg_evidence_attention': ['mean', 'std', 'min', 'max'],
                        'max_head': ['mean', 'std', 'min', 'max']
                    }).round(4)
                    st.dataframe(domain_stats)
                    
                    fig, ax = plt.subplots()
                    domain_means = df.groupby('domain')['avg_evidence_attention'].mean().sort_values(ascending=False)
                    ax.bar(domain_means.index, domain_means.values)
                    ax.set_title('Domain Performance by Average Evidence Attention')
                    ax.set_ylabel('Average Evidence Attention')
                    st.pyplot(fig)

                # 헤드 분포 분석
                with st.expander("🧠 헤드 분포 분석", expanded=False):
                    head_counts = df['max_head'].value_counts().sort_index()
                    st.bar_chart(head_counts)
                    
                    # 도메인-헤드 히트맵
                    hist = df.groupby(["domain", "max_head"]).size().unstack(fill_value=0)
                    fig2, ax2 = plt.subplots(figsize=(min(12, 2+0.5*hist.shape[1]), 1.5+0.5*hist.shape[0]))
                    sns.heatmap(hist, annot=True, fmt="d", cmap="YlOrRd", ax=ax2)
                    ax2.set_xlabel("Head Index")
                    ax2.set_ylabel("Domain")
                    ax2.set_title("Domain-Head Evidence Response Frequency Heatmap")
                    st.pyplot(fig2)

                # 어텐션 패턴 분석
                with st.expander("📈 Evidence Attention 분포", expanded=False):
                    fig3, ax3 = plt.subplots()
                    ax3.hist(df['avg_evidence_attention'], bins=30, alpha=0.7)
                    ax3.set_title('Overall Evidence Attention Distribution')
                    ax3.set_xlabel('Evidence Attention')
                    ax3.set_ylabel('Frequency')
                    st.pyplot(fig3)
                    
                    # 도메인별 분포
                    for domain in df['domain'].unique():
                        st.markdown(f"**{domain} 도메인 Evidence Attention 분포**")
                        fig4, ax4 = plt.subplots()
                        ax4.hist(df[df['domain']==domain]['avg_evidence_attention'], bins=20, alpha=0.7)
                        ax4.set_title(f'{domain} Domain Evidence Attention Distribution')
                        ax4.set_xlabel('Evidence Attention')
                        ax4.set_ylabel('Frequency')
                        st.pyplot(fig4)

                # 프롬프트 길이 영향 분석
                with st.expander("📝 프롬프트 길이 영향 분석", expanded=False):
                    df['prompt_length'] = df['prompt'].str.len()
                    df['token_count'] = df['tokens'].apply(len)
                    fig5, (ax5, ax6) = plt.subplots(1,2,figsize=(12,5))
                    ax5.scatter(df['prompt_length'], df['avg_evidence_attention'], alpha=0.6)
                    ax5.set_xlabel('Prompt Length (characters)')
                    ax5.set_ylabel('Evidence Attention')
                    ax5.set_title('Prompt Length vs Evidence Attention')
                    ax6.scatter(df['token_count'], df['avg_evidence_attention'], alpha=0.6)
                    ax6.set_xlabel('Token Count')
                    ax6.set_ylabel('Evidence Attention')
                    ax6.set_title('Token Count vs Evidence Attention')
                    st.pyplot(fig5)

                # Evidence 인덱스 패턴 분석
                with st.expander("🧩 Evidence 인덱스 패턴 분석", expanded=False):
                    evidence_counts = df['evidence_indices'].apply(len)
                    st.markdown(f"평균 evidence 토큰 수: {evidence_counts.mean():.2f}")
                    fig7, ax7 = plt.subplots()
                    ax7.hist(evidence_counts, bins=20, alpha=0.7)
                    ax7.set_title('Evidence Token Count Distribution')
                    ax7.set_xlabel('Evidence Token Count')
                    ax7.set_ylabel('Frequency')
                    st.pyplot(fig7)

        with tab3:
            st.markdown("### 📈 n차 실험 결과 비교 분석")
            st.markdown("여러 실험 결과를 선택하여 비교 분석을 수행할 수 있습니다.")
            
            # 실험 결과 파일 다중 선택
            selected_results = st.multiselect(
                "비교할 실험 결과 파일들을 선택하세요 (최대 5개)",
                result_files,
                default=result_files[:2] if len(result_files) >= 2 else result_files
            )
            
            if len(selected_results) >= 2:
                if st.button("n차 실험 비교 분석 실행", key="n_analysis"):
                    # 선택된 실험 결과들을 로드
                    experiment_data = {}
                    for filename in selected_results:
                        results = load_experiment_result(filename)
                        df = pd.DataFrame(results)
                        # 파일명에서 실험 정보 추출
                        experiment_name = filename.replace('.json', '').replace('_', ' ')
                        experiment_data[experiment_name] = df
                    
                    st.markdown("### 📊 실험 개요 비교")
                    
                    # 기본 통계 비교
                    comparison_data = []
                    for exp_name, df in experiment_data.items():
                        # 모델 정보 추출 (새로운 실험 결과에는 model_name이 포함됨)
                        model_info = "알 수 없음"
                        if 'model_name' in df.columns and len(df) > 0:
                            model_info = df['model_name'].iloc[0] if df['model_name'].iloc[0] else "알 수 없음"
                        
                        comparison_data.append({
                            "실험명": exp_name,
                            "모델": model_info,
                            "총 샘플 수": len(df),
                            "도메인 수": len(df['domain'].unique()),
                            "평균 Evidence Attention": round(df['avg_evidence_attention'].mean(), 4),
                            "최대 Evidence Attention": round(df['avg_evidence_attention'].max(), 4),
                            "가장 많이 사용된 헤드": f"헤드 {df['max_head'].value_counts().idxmax()}",
                            "헤드 다양성": len(df['max_head'].unique())
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # 모델별 토큰화 차이 경고
                    unique_models = comparison_df['모델'].unique()
                    if len(unique_models) > 1 and not all(model == "알 수 없음" for model in unique_models):
                        st.warning("""
                        ⚠️ **모델별 토큰화 차이 주의사항**
                        
                        선택된 실험들이 서로 다른 모델을 사용하고 있습니다. 각 모델은 서로 다른 토크나이저를 사용하므로:
                        - **토큰화 결과가 다릅니다**: 같은 텍스트라도 모델마다 다른 토큰으로 분할됩니다.
                        - **Evidence 인덱스가 다릅니다**: 토큰화가 다르므로 evidence 토큰의 위치(인덱스)도 달라집니다.
                        - **헤드 비교에 주의가 필요합니다**: 토큰화 차이로 인해 같은 헤드라도 실제로는 다른 토큰을 보고 있을 수 있습니다.
                        
                        이러한 차이를 고려하여 결과를 해석해주세요.
                        """)
                    
                    # 모델별 그룹화 분석
                    if len(unique_models) > 1 and not all(model == "알 수 없음" for model in unique_models):
                        st.markdown("### 🤖 모델별 분석")
                        
                        for model in unique_models:
                            if model != "알 수 없음":
                                st.markdown(f"**{model} 모델 실험들:**")
                                model_experiments = [exp for exp, df in experiment_data.items() 
                                                   if 'model_name' in df.columns and len(df) > 0 and df['model_name'].iloc[0] == model]
                                
                                if model_experiments:
                                    # 해당 모델의 실험들만 필터링
                                    model_data = {exp: df for exp, df in experiment_data.items() if exp in model_experiments}
                                    
                                    # 모델별 통계
                                    model_stats = []
                                    for exp_name, df in model_data.items():
                                        model_stats.append({
                                            "실험명": exp_name,
                                            "평균 Evidence Attention": round(df['avg_evidence_attention'].mean(), 4),
                                            "가장 많이 사용된 헤드": f"헤드 {df['max_head'].value_counts().idxmax()}",
                                            "헤드 다양성": len(df['max_head'].unique())
                                        })
                                    
                                    st.dataframe(pd.DataFrame(model_stats))
                                    
                                    # 모델별 헤드 사용 패턴
                                    fig_model, ax_model = plt.subplots(figsize=(12, 6))
                                    for exp_name, df in model_data.items():
                                        head_counts = df['max_head'].value_counts().sort_index()
                                        ax_model.plot(head_counts.index, head_counts.values, 
                                                    marker='o', label=exp_name, linewidth=2, markersize=6)
                                    
                                    ax_model.set_xlabel('Head Index')
                                    ax_model.set_ylabel('Usage Frequency')
                                    ax_model.set_title(f'{model} Model - Head Usage Pattern Comparison')
                                    ax_model.legend()
                                    ax_model.grid(True, alpha=0.3)
                                    st.pyplot(fig_model)
                    
                    # 토큰화 차이 분석 (새로운 실험 결과에만 해당)
                    st.markdown("### 🔤 토큰화 차이 분석")
                    
                    # 토크나이저 정보가 있는 실험들만 필터링
                    experiments_with_tokenizer = []
                    for exp_name, df in experiment_data.items():
                        if 'tokenizer_name' in df.columns and len(df) > 0 and df['tokenizer_name'].iloc[0] != "unknown":
                            experiments_with_tokenizer.append(exp_name)
                    
                    if len(experiments_with_tokenizer) >= 2:
                        st.markdown("**토크나이저 정보가 있는 실험들 간의 비교:**")
                        
                        # 같은 프롬프트에 대한 토큰화 차이 분석
                        tokenizer_comparison = {}
                        for exp_name in experiments_with_tokenizer:
                            df = experiment_data[exp_name]
                            if len(df) > 0:
                                tokenizer_name = df['tokenizer_name'].iloc[0]
                                tokenizer_comparison[exp_name] = {
                                    'tokenizer': tokenizer_name,
                                    'avg_tokens_per_prompt': df['tokens'].apply(len).mean(),
                                    'max_tokens_per_prompt': df['tokens'].apply(len).max(),
                                    'min_tokens_per_prompt': df['tokens'].apply(len).min()
                                }
                        
                        if tokenizer_comparison:
                            tokenizer_df = pd.DataFrame(tokenizer_comparison).T
                            st.dataframe(tokenizer_df)
                            
                            # 토큰 수 분포 비교
                            fig_tokens, ax_tokens = plt.subplots(figsize=(12, 6))
                            for exp_name in experiments_with_tokenizer:
                                df = experiment_data[exp_name]
                                token_counts = df['tokens'].apply(len)
                                ax_tokens.hist(token_counts, bins=20, alpha=0.6, label=f"{exp_name} ({df['tokenizer_name'].iloc[0]})")
                            
                            ax_tokens.set_xlabel('Token Count per Prompt')
                            ax_tokens.set_ylabel('Frequency')
                            ax_tokens.set_title('Token Count Distribution by Tokenizer')
                            ax_tokens.legend()
                            ax_tokens.grid(True, alpha=0.3)
                            st.pyplot(fig_tokens)
                    else:
                        st.info("토크나이저 정보가 있는 실험이 충분하지 않아 토큰화 차이 분석을 수행할 수 없습니다.")
                    
                    # 도메인별 성능 비교
                    st.markdown("### 📈 도메인별 Evidence Attention 비교")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for i, (exp_name, df) in enumerate(experiment_data.items()):
                        domain_means = df.groupby('domain')['avg_evidence_attention'].mean()
                        ax.plot(domain_means.index, domain_means.values, 
                               marker='o', label=exp_name, linewidth=2, markersize=8)
                    
                    ax.set_xlabel('Domain')
                    ax.set_ylabel('Average Evidence Attention')
                    ax.set_title('Domain Performance Comparison Across Experiments')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # 도메인-헤드 히트맵 비교
                    st.markdown("### 🔥 도메인-헤드 히트맵 비교")
                    
                    # 모든 실험의 데이터를 통합하여 하나의 히트맵으로 표시
                    combined_data = []
                    for exp_name, df in experiment_data.items():
                        for _, row in df.iterrows():
                            combined_data.append({
                                'experiment': exp_name,
                                'domain': row['domain'],
                                'max_head': row['max_head'],
                                'avg_evidence_attention': row['avg_evidence_attention']
                            })
                    
                    combined_df = pd.DataFrame(combined_data)
                    
                    # 실험별 도메인-헤드 히트맵
                    fig_combined, ax_combined = plt.subplots(figsize=(15, 8))
                    
                    # 실험별로 다른 색상 사용
                    pivot_data = combined_df.groupby(['experiment', 'domain', 'max_head']).size().unstack(fill_value=0)
                    
                    # 히트맵 생성
                    sns.heatmap(pivot_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax_combined, cbar_kws={'label': 'Frequency'})
                    ax_combined.set_xlabel("Head Index")
                    ax_combined.set_ylabel("Experiment-Domain")
                    ax_combined.set_title("Combined Domain-Head Evidence Response Frequency Across Experiments")
                    
                    # y축 라벨을 더 읽기 쉽게 조정
                    y_labels = [f"{exp}-{domain}" for exp, domain in pivot_data.index]
                    ax_combined.set_yticklabels(y_labels, rotation=0)
                    
                    st.pyplot(fig_combined)
                    st.caption("*모든 실험의 도메인-헤드 반응 빈도를 하나의 히트맵으로 통합 표시*")
                    
                    # 실험별 평균 Evidence Attention 비교 히트맵
                    st.markdown("### 📊 실험별 평균 Evidence Attention 비교")
                    
                    # 실험-도메인별 평균 Evidence Attention 계산
                    avg_attention_pivot = combined_df.groupby(['experiment', 'domain'])['avg_evidence_attention'].mean().unstack(fill_value=0)
                    
                    fig_avg, ax_avg = plt.subplots(figsize=(12, 6))
                    sns.heatmap(avg_attention_pivot, annot=True, fmt=".4f", cmap="Blues", ax=ax_avg, cbar_kws={'label': 'Average Evidence Attention'})
                    ax_avg.set_xlabel("Domain")
                    ax_avg.set_ylabel("Experiment")
                    ax_avg.set_title("Average Evidence Attention by Experiment and Domain")
                    st.pyplot(fig_avg)
                    st.caption("*각 실험-도메인 조합의 평균 Evidence Attention 값*")
                    
                    # 헤드 사용 패턴 통합 비교
                    st.markdown("### 🧠 헤드 사용 패턴 통합 비교")
                    
                    # 모든 실험의 헤드 사용 빈도를 통합
                    all_head_usage = combined_df['max_head'].value_counts().sort_index()
                    
                    fig_head, ax_head = plt.subplots(figsize=(12, 6))
                    bars = ax_head.bar(all_head_usage.index, all_head_usage.values, alpha=0.7)
                    ax_head.set_xlabel('Head Index')
                    ax_head.set_ylabel('Total Usage Frequency')
                    ax_head.set_title('Combined Head Usage Pattern Across All Experiments')
                    ax_head.grid(True, alpha=0.3)
                    
                    # 상위 5개 헤드 강조
                    top_heads = all_head_usage.head(5)
                    for head in top_heads.index:
                        if head in all_head_usage.index:
                            idx = list(all_head_usage.index).index(head)
                            bars[idx].set_color('red')
                            bars[idx].set_alpha(0.8)
                    
                    st.pyplot(fig_head)
                    st.caption("*모든 실험을 통합한 헤드 사용 빈도 (빨간색: 상위 5개 헤드)*")
                    
                    # 실험별 헤드 사용 패턴 비교 (하나의 그래프)
                    st.markdown("### 📈 실험별 헤드 사용 패턴 비교")
                    
                    fig_compare, ax_compare = plt.subplots(figsize=(15, 8))
                    
                    # 각 실험별로 헤드 사용 빈도 계산
                    for exp_name, df in experiment_data.items():
                        head_counts = df['max_head'].value_counts().sort_index()
                        ax_compare.plot(head_counts.index, head_counts.values, 
                                      marker='o', label=exp_name, linewidth=2, markersize=6)
                    
                    ax_compare.set_xlabel('Head Index')
                    ax_compare.set_ylabel('Usage Frequency')
                    ax_compare.set_title('Head Usage Pattern Comparison Across Experiments')
                    ax_compare.legend()
                    ax_compare.grid(True, alpha=0.3)
                    st.pyplot(fig_compare)
                    st.caption("*각 실험에서 사용된 헤드의 빈도를 선그래프로 비교*")
                    
                    # Evidence Attention 분포 통합 비교
                    st.markdown("### 📊 Evidence Attention 분포 통합 비교")
                    
                    fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
                    
                    # 각 실험별로 히스토그램 그리기
                    for exp_name, df in experiment_data.items():
                        ax_dist.hist(df['avg_evidence_attention'], bins=20, alpha=0.6, label=exp_name, density=True)
                    
                    ax_dist.set_xlabel('Evidence Attention')
                    ax_dist.set_ylabel('Density')
                    ax_dist.set_title('Evidence Attention Distribution Comparison (Normalized)')
                    ax_dist.legend()
                    ax_dist.grid(True, alpha=0.3)
                    st.pyplot(fig_dist)
                    st.caption("*각 실험의 Evidence Attention 분포를 정규화하여 비교*")
                    
                    # 통계적 유의성 검정
                    st.markdown("### 📊 통계적 비교")
                    
                    if len(experiment_data) == 2:
                        # 두 실험 결과의 Evidence Attention 비교
                        exp_names = list(experiment_data.keys())
                        df1 = experiment_data[exp_names[0]]
                        df2 = experiment_data[exp_names[1]]
                        
                        # t-test 수행
                        t_stat, p_value = stats.ttest_ind(df1['avg_evidence_attention'], df2['avg_evidence_attention'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("t-통계량", f"{t_stat:.4f}")
                        with col2:
                            st.metric("p-값", f"{p_value:.4f}")
                        with col3:
                            significance = "유의함" if p_value < 0.05 else "유의하지 않음"
                            st.metric("통계적 유의성", significance)
                        
                        st.markdown(f"""
                        **해석:**
                        - p-값이 0.05보다 작으면 두 실험 결과 간에 통계적으로 유의한 차이가 있다고 볼 수 있습니다.
                        - 현재 p-값: {p_value:.4f} ({significance})
                        """)
                    elif len(experiment_data) > 2:
                        # 다중 실험 비교를 위한 ANOVA
                        # ANOVA 수행
                        groups = [df['avg_evidence_attention'].values for df in experiment_data.values()]
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("F-통계량", f"{f_stat:.4f}")
                        with col2:
                            st.metric("p-값", f"{p_value:.4f}")
                        with col3:
                            significance = "유의함" if p_value < 0.05 else "유의하지 않음"
                            st.metric("통계적 유의성", significance)
                        
                        st.markdown(f"""
                        **해석:**
                        - ANOVA를 통해 여러 실험 결과 간의 통계적 차이를 검정합니다.
                        - p-값이 0.05보다 작으면 실험들 간에 통계적으로 유의한 차이가 있다고 볼 수 있습니다.
                        - 현재 p-값: {p_value:.4f} ({significance})
                        """)
                        
                    # 실험 간 일관성 분석
                    st.markdown("### 🔄 실험 간 일관성 분석")
                    
                    # 공통 헤드 분석
                    all_heads = set()
                    for df in experiment_data.values():
                        all_heads.update(df['max_head'].unique())
                    
                    head_consistency = {}
                    for head in all_heads:
                        consistency_count = 0
                        for df in experiment_data.values():
                            if head in df['max_head'].values:
                                consistency_count += 1
                        head_consistency[head] = consistency_count / len(experiment_data)
                    
                    # 일관성 높은 헤드들 표시
                    consistent_heads = {head: score for head, score in head_consistency.items() if score >= 0.5}
                    
                    if consistent_heads:
                        st.markdown("**여러 실험에서 일관되게 사용된 헤드들 (50% 이상):**")
                        for head, consistency in sorted(consistent_heads.items(), key=lambda x: x[1], reverse=True):
                            st.markdown(f"- 헤드 {head}: {consistency*100:.1f}% 일관성")
                    else:
                        st.markdown("**일관되게 사용된 헤드가 없습니다.**")
                    
                    # 일관성 시각화
                    if head_consistency:
                        fig_consistency, ax_consistency = plt.subplots(figsize=(12, 6))
                        heads = list(head_consistency.keys())
                        consistency_scores = list(head_consistency.values())
                        
                        bars = ax_consistency.bar(heads, consistency_scores, alpha=0.7)
                        ax_consistency.set_xlabel('Head Index')
                        ax_consistency.set_ylabel('Consistency Score')
                        ax_consistency.set_title('Head Consistency Across Experiments')
                        ax_consistency.grid(True, alpha=0.3)
                        
                        # 50% 이상 일관성 있는 헤드 강조
                        for i, score in enumerate(consistency_scores):
                            if score >= 0.5:
                                bars[i].set_color('green')
                                bars[i].set_alpha(0.8)
                        
                        st.pyplot(fig_consistency)
                        st.caption("*각 헤드의 실험 간 일관성 점수 (녹색: 50% 이상 일관성)*")
                    
                    # 종합 분석 요약
                    st.markdown("### 📋 종합 분석 요약")
                    
                    summary_points = []
                    
                    # 가장 성능이 좋은 실험
                    best_exp = max(experiment_data.items(), key=lambda x: x[1]['avg_evidence_attention'].mean())
                    summary_points.append(f"**최고 성능 실험:** {best_exp[0]} (평균 Evidence Attention: {best_exp[1]['avg_evidence_attention'].mean():.4f})")
                    
                    # 가장 일관된 헤드
                    if consistent_heads:
                        most_consistent_head = max(consistent_heads.items(), key=lambda x: x[1])
                        summary_points.append(f"**가장 일관된 헤드:** 헤드 {most_consistent_head[0]} ({most_consistent_head[1]*100:.1f}% 일관성)")
                    
                    # 도메인별 패턴
                    domain_patterns = {}
                    for exp_name, df in experiment_data.items():
                        for domain in df['domain'].unique():
                            if domain not in domain_patterns:
                                domain_patterns[domain] = []
                            domain_patterns[domain].append(df[df['domain']==domain]['avg_evidence_attention'].mean())
                    
                    for domain, values in domain_patterns.items():
                        if len(values) > 1:
                            variance = np.var(values)
                            if variance < 0.01:  # 낮은 분산
                                summary_points.append(f"**{domain} 도메인:** 실험 간 일관된 성능 (분산: {variance:.4f})")
                            else:
                                summary_points.append(f"**{domain} 도메인:** 실험 간 성능 변동 있음 (분산: {variance:.4f})")
                    
                    # 전체 통계
                    total_samples = sum(len(df) for df in experiment_data.values())
                    summary_points.append(f"**총 분석 샘플 수:** {total_samples}")
                    summary_points.append(f"**분석 실험 수:** {len(experiment_data)}")
                    
                    for point in summary_points:
                        st.markdown(f"- {point}")
                    
                    st.success("n차 실험 비교 분석이 완료되었습니다!")
            else:
                st.warning("비교 분석을 위해서는 최소 2개의 실험 결과를 선택해주세요.")
    else:
        st.warning("분석할 실험 결과 파일이 없습니다. 실험 탭에서 먼저 실험을 수행해주세요.") 