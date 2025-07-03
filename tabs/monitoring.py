import streamlit as st
import psutil
import subprocess
import time

def get_gpu_info():
    """NVIDIA GPU 정보 가져오기 (캐시된 경우 사용)"""
    # 세션 상태에서 캐시된 GPU 정보 확인
    if "gpu_info_cache" in st.session_state and "gpu_cache_time" in st.session_state:
        # 캐시가 5초 이내인 경우 사용
        if time.time() - st.session_state.gpu_cache_time < 5:
            return st.session_state.gpu_info_cache
    
    try:
        # nvidia-smi 명령어 실행
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, 
                              text=True)
        if result.returncode == 0:
            # GPU 정보 파싱
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                name, total, used, free, temp, util = line.split(', ')
                # MiB를 GiB로 변환 (1 GiB = 1024 MiB)
                total = float(total) / 1024
                used = float(used) / 1024
                free = float(free) / 1024
                gpu_info.append({
                    'name': name,
                    'total_memory': total,
                    'used_memory': used,
                    'free_memory': free,
                    'temperature': float(temp),
                    'utilization': float(util)
                })
            
            # 세션 상태에 캐시
            st.session_state.gpu_info_cache = gpu_info
            st.session_state.gpu_cache_time = time.time()
            return gpu_info
        return None
    except Exception:
        return None

def get_system_memory():
    """시스템 메모리 정보 가져오기 (캐시된 경우 사용)"""
    # 세션 상태에서 캐시된 메모리 정보 확인
    if "memory_info_cache" in st.session_state and "memory_cache_time" in st.session_state:
        # 캐시가 3초 이내인 경우 사용
        if time.time() - st.session_state.memory_cache_time < 3:
            return st.session_state.memory_info_cache
    
    memory = psutil.virtual_memory()
    # Bytes를 GiB로 변환 (1 GiB = 1024^3 bytes)
    gib = 1024 * 1024 * 1024
    memory_info = {
        'total': memory.total / gib,
        'used': memory.used / gib,  # 캐시/버퍼를 포함한 사용 중인 메모리
        'available': memory.available / gib,  # 실제 사용 가능한 메모리
        'free': memory.free / gib,  # 완전히 비어있는 메모리
        'cached': (memory.cached if hasattr(memory, 'cached') else 0) / gib,  # 캐시 메모리
        'buffers': (memory.buffers if hasattr(memory, 'buffers') else 0) / gib  # 버퍼 메모리
    }
    
    # 세션 상태에 캐시
    st.session_state.memory_info_cache = memory_info
    st.session_state.memory_cache_time = time.time()
    return memory_info

def format_size(size_gb):
    """GiB 단위를 보기 좋게 포맷팅"""
    if size_gb >= 1024:
        return f"{size_gb/1024:.1f} TiB"
    return f"{size_gb:.1f} GiB"

def show():
    st.title("🖥️ WSL 시스템 모니터링")
    
    # 세션 상태 초기화
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    st.info("""
    ℹ️ 현재 표시되는 정보는 WSL(Windows Subsystem for Linux) 환경에 할당된 리소스입니다.
    - 메모리: WSL에 할당된 가상 메모리
    - GPU: WSL에서 접근 가능한 GPU
    """)
    
    # 자동 새로고침 설정
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_refresh = st.checkbox("자동 새로고침", value=st.session_state.auto_refresh, key="auto_refresh_checkbox")
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            # st.rerun() 제거 - 페이지 새로고침 방지
    
    with col2:
        if st.button("🔄 수동 새로고침", key="manual_refresh"):
            # 캐시 초기화
            cache_keys = ['gpu_info_cache', 'gpu_cache_time', 'memory_info_cache', 'memory_cache_time']
            for key in cache_keys:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("새로고침되었습니다!")
            # st.rerun() 제거 - 페이지 새로고침 방지
    
    # GPU 정보 표시
    st.subheader("🎮 WSL GPU 상태")
    gpu_info = get_gpu_info()
    
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            with st.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### GPU {i}: {gpu['name']}")
                    
                    # GPU 메모리 사용량 차트
                    memory_data = {
                        'Memory (GiB)': [gpu['used_memory'], gpu['free_memory']],
                        'Type': ['사용 중', '여유']
                    }
                    st.bar_chart(memory_data, y='Memory (GiB)')
                
                with col2:
                    # GPU 상세 정보
                    st.metric("온도", f"{gpu['temperature']}°C")
                    st.metric("사용률", f"{gpu['utilization']:.1f}%")
                    st.metric("총 메모리", format_size(gpu['total_memory']))
                    st.metric("사용 중인 메모리", format_size(gpu['used_memory']))
                    st.metric("여유 메모리", format_size(gpu['free_memory']))
    else:
        st.warning("⚠️ WSL에서 NVIDIA GPU를 찾을 수 없거나 nvidia-smi 명령어를 사용할 수 없습니다.")
    
    # 시스템 메모리 정보 표시
    st.subheader("💾 WSL 메모리")
    memory_info = get_system_memory()
    
    # 메모리 사용량 차트 (캐시/버퍼 포함)
    memory_data = {
        'Memory (GiB)': [
            memory_info['used'] - memory_info['cached'] - memory_info['buffers'],  # 실제 사용
            memory_info['cached'] + memory_info['buffers'],  # 캐시/버퍼
            memory_info['available']  # 사용 가능
        ],
        'Type': ['실제 사용', '캐시/버퍼', '사용 가능']
    }
    st.bar_chart(memory_data, y='Memory (GiB)')
    
    # 메모리 상세 정보
    col1, col2 = st.columns(2)
    with col1:
        st.metric("WSL 총 할당 메모리", format_size(memory_info['total']))
        st.metric("실제 사용 중인 메모리", format_size(memory_info['used'] - memory_info['cached'] - memory_info['buffers']))
        st.metric("사용 가능한 메모리", format_size(memory_info['available']))
    with col2:
        st.metric("캐시 메모리", format_size(memory_info['cached']))
        st.metric("버퍼 메모리", format_size(memory_info['buffers']))
        st.metric("완전히 비어있는 메모리", format_size(memory_info['free']))
    
    # WSL 메모리 설정 안내
    st.markdown("""
    ---
    ### ℹ️ WSL 메모리 설정 안내
    
    WSL의 메모리 할당을 조정하려면 Windows의 `.wslconfig` 파일을 수정하세요:
    
    1. `%UserProfile%\.wslconfig` 파일을 생성 또는 편집
    2. 다음과 같이 메모리 설정을 추가:
    ```ini
    [wsl2]
    memory=8GB   # WSL에 할당할 메모리
    swap=2GB     # 스왑 메모리 크기
    ```
    3. WSL을 재시작: `wsl --shutdown` 후 다시 시작
    """)
    
    # 자동 새로고침 처리
    if st.session_state.auto_refresh:
        time.sleep(2)  # 2초 대기
        # st.rerun() 제거 - 페이지 새로고침 방지
    
    # 캐시 초기화 버튼 (디버깅용)
    if st.sidebar.button("🗑️ 캐시 초기화", help="모든 캐시를 초기화합니다", key="clear_cache_monitoring"):
        # GPU 및 메모리 정보 캐시 초기화
        cache_keys = ['gpu_info_cache', 'gpu_cache_time', 'memory_info_cache', 'memory_cache_time', 'auto_refresh']
        for key in cache_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("캐시가 초기화되었습니다!")
        # st.rerun() 제거 - 페이지 새로고침 방지 