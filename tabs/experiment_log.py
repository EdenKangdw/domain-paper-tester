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
    result_files = list_experiment_results()
    if result_files:
        selected_result = st.selectbox("조회할 실험 결과 파일을 선택하세요", result_files)
        if st.button("선택한 결과 불러오기"):
            loaded_results = load_experiment_result(selected_result)
            df = pd.DataFrame(loaded_results)
            st.dataframe(df[["domain", "max_head", "avg_evidence_attention"]])
            st.markdown("#### 도메인별 evidence에 가장 많이 반응한 헤드 분포")
            hist = df.groupby(["domain", "max_head"]).size().unstack(fill_value=0)
            st.bar_chart(hist)
            st.caption("*도메인별로 evidence에 가장 강하게 반응한 헤드의 빈도 분포*")
            fig, ax = plt.subplots(figsize=(min(12, 2+0.5*hist.shape[1]), 1.5+0.5*hist.shape[0]))
            sns.heatmap(hist, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_xlabel("헤드 인덱스")
            ax.set_ylabel("도메인")
            ax.set_title("도메인-헤드별 evidence 반응 빈도 히트맵")
            st.pyplot(fig)
            st.caption("*x축: 헤드 인덱스, y축: 도메인, 값: evidence에 가장 많이 반응한 횟수*")
    else:
        st.info("저장된 실험 결과 파일이 없습니다.") 