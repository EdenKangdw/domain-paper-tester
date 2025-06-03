import streamlit as st
import psutil
import subprocess

def get_gpu_info():
    """NVIDIA GPU 정보 가져오기"""
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
            return gpu_info
        return None
    except Exception:
        return None

def get_system_memory():
    """시스템 메모리 정보 가져오기"""
    memory = psutil.virtual_memory()
    # Bytes를 GiB로 변환 (1 GiB = 1024^3 bytes)
    gib = 1024 * 1024 * 1024
    return {
        'total': memory.total / gib,
        'used': memory.used / gib,  # 캐시/버퍼를 포함한 사용 중인 메모리
        'available': memory.available / gib,  # 실제 사용 가능한 메모리
        'free': memory.free / gib,  # 완전히 비어있는 메모리
        'cached': (memory.cached if hasattr(memory, 'cached') else 0) / gib,  # 캐시 메모리
        'buffers': (memory.buffers if hasattr(memory, 'buffers') else 0) / gib  # 버퍼 메모리
    }

def format_size(size_gb):
    """GiB 단위를 보기 좋게 포맷팅"""
    if size_gb >= 1024:
        return f"{size_gb/1024:.1f} TiB"
    return f"{size_gb:.1f} GiB"

def show():
    st.title("🖥️ WSL 시스템 모니터링")
    
    st.info("""
    ℹ️ 현재 표시되는 정보는 WSL(Windows Subsystem for Linux) 환경에 할당된 리소스입니다.
    - 메모리: WSL에 할당된 가상 메모리
    - GPU: WSL에서 접근 가능한 GPU
    """)
    
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
    
    # 자동 새로고침
    if st.button("🔄 새로고침"):
        st.experimental_rerun() 