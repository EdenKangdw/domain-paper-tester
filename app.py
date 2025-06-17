import streamlit as st
from tabs import experiment, experiment_log, dataset_generator, model_loader, monitoring
import os

# 페이지 설정
st.set_page_config(
    page_title="Attention Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바에 탭 선택 UI 추가
st.sidebar.title("🧠 Attention Analysis")

# 현재 선택된 탭을 URL 파라미터에서 가져오기
current_tab = st.query_params.get("tab", "model_loader")

# 탭 순서 지정
ordered_tabs = [
    ("model_loader", "🤖 모델 로드"),
    ("dataset_generator", "📚 데이터셋 생성"),
    ("experiment", "🔬 실험"),
    ("experiment_log", "📊 실험기록"),
    ("monitoring", "📈 모니터링")
]

# 탭 선택 버튼들 생성
for tab_id, tab_name in ordered_tabs:
    if st.sidebar.button(
        tab_name,
        use_container_width=True,
        type="primary" if current_tab == tab_id else "secondary"
    ):
        st.query_params["tab"] = tab_id
        st.rerun()

# 선택된 탭에 따라 해당 페이지 표시
if current_tab == "experiment":
    experiment.show()
elif current_tab == "experiment_log":
    experiment_log.show()
elif current_tab == "dataset_generator":
    dataset_generator.show()
elif current_tab == "model_loader":
    model_loader.show()
elif current_tab == "monitoring":
    monitoring.show()
