import streamlit as st
from pathlib import Path
import json
import time
from datetime import datetime

def show():
    """
    페이지 메인 함수 - 모든 페이지에서 사용할 수 있는 템플릿
    
    주요 가이드라인:
    1. st.rerun() 사용 금지 - 페이지 새로고침 방지
    2. 모든 버튼에 고유한 key 지정
    3. 세션 상태 관리를 통한 부드러운 상태 업데이트
    4. 캐시 활용으로 성능 최적화
    """
    
    st.title("📄 페이지 제목")
    
    # 세션 상태 초기화
    if 'page_initialized' not in st.session_state:
        st.session_state.page_initialized = True
        st.session_state.counter = 0
        # 기타 필요한 세션 상태들...
    
    # 섹션 1: 기본 설정
    st.subheader("⚙️ 기본 설정")
    
    # 설정 옵션들
    option1 = st.selectbox(
        "옵션 1",
        ["선택 1", "선택 2", "선택 3"],
        key="option1_selector"
    )
    
    option2 = st.number_input(
        "옵션 2",
        min_value=1,
        max_value=100,
        value=10,
        key="option2_input"
    )
    
    # 섹션 2: 액션 버튼들
    st.subheader("🎯 액션")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 로드", type="primary", key="load_btn"):
            # 로드 로직
            st.session_state.counter += 1
            st.success("로드 완료!")
            # st.rerun() 사용 금지
    
    with col2:
        if st.button("🔄 새로고침", type="secondary", key="refresh_btn"):
            # 새로고침 로직
            cache_keys = ['cache1', 'cache2']  # 캐시 키들
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("새로고침 완료!")
            # st.rerun() 사용 금지
    
    with col3:
        if st.button("📤 내보내기", type="secondary", key="export_btn"):
            # 내보내기 로직
            st.info("내보내기 완료!")
            # st.rerun() 사용 금지
    
    # 섹션 3: 결과 표시
    st.subheader("📊 결과")
    
    if st.session_state.counter > 0:
        st.write(f"처리된 항목: {st.session_state.counter}개")
    
    # 캐시 활용 예시
    cache_key = f"data_cache_{option1}_{option2}"
    if cache_key not in st.session_state:
        # 캐시에 없는 경우 계산
        st.session_state[cache_key] = f"계산된 데이터: {option1} + {option2}"
    
    st.write(f"캐시된 데이터: {st.session_state[cache_key]}")
    
    # 섹션 4: 디버깅 도구
    st.subheader("🔧 디버깅")
    
    # 캐시 초기화 버튼 (모든 페이지에 포함)
    if st.sidebar.button("🗑️ 캐시 초기화", help="모든 캐시를 초기화합니다", key="clear_cache_page"):
        # 현재 페이지 관련 캐시만 초기화
        keys_to_remove = [key for key in st.session_state.keys() 
                         if key.startswith(('cache_', 'data_cache_')) or 
                            key in ['page_initialized', 'counter']]
        for key in keys_to_remove:
            del st.session_state[key]
        st.sidebar.success("캐시가 초기화되었습니다!")
        # st.rerun() 사용 금지
    
    # 세션 상태 디버깅 (개발 중에만 사용)
    if st.sidebar.checkbox("디버깅 모드", key="debug_mode"):
        st.sidebar.write("현재 세션 상태:")
        st.sidebar.json({k: str(v)[:50] + "..." if len(str(v)) > 50 else str(v) 
                        for k, v in st.session_state.items() 
                        if not k.startswith('_')})

# 사용 예시:
# 1. 이 템플릿을 복사하여 새 페이지 생성
# 2. show() 함수 내용을 수정
# 3. st.rerun() 사용 금지
# 4. 모든 버튼에 고유한 key 지정
# 5. 세션 상태를 활용한 부드러운 상태 관리 