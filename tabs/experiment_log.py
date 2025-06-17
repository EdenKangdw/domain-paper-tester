import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        """)
    
    result_files = list_experiment_results()
    if result_files:
        selected_result = st.selectbox("조회할 실험 결과 파일을 선택하세요", result_files)
        if st.button("선택한 결과 불러오기"):
            loaded_results = load_experiment_result(selected_result)
            df = pd.DataFrame(loaded_results)
            
            # 기본 데이터프레임 표시
            st.markdown("### 📊 기본 실험 결과")
            st.dataframe(df[["domain", "max_head", "avg_evidence_attention"]])
            
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
            
            # 추가 분석 제안
            st.markdown("""
            ### 💡 추가 분석 제안
            1. **도메인별 평균 어텐션 강도:**
               - 각 도메인에서 evidence에 대한 평균 어텐션 강도를 비교하여, 어떤 도메인이 더 강한 evidence 반응을 보이는지 분석할 수 있습니다.
            
            2. **헤드 그룹 분석:**
               - 비슷한 패턴을 보이는 헤드들을 그룹화하여, 각 그룹이 어떤 종류의 evidence 처리에 특화되어 있는지 분석할 수 있습니다.
            
            3. **프롬프트별 상세 분석:**
               - 특정 도메인이나 헤드에서 특이한 패턴이 발견되면, 해당하는 개별 프롬프트들을 자세히 분석하여 패턴의 원인을 파악할 수 있습니다.
            """)
    else:
        st.info("저장된 실험 결과 파일이 없습니다.") 