#!/usr/bin/env python3
"""
Evidence 추출에서 모델별 경로가 제대로 작동하는지 테스트하는 스크립트
"""

import sys
from pathlib import Path

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

from evidence_extractor import load_origin_prompts, get_available_files, load_selected_files

def test_model_paths():
    """모델별 경로가 제대로 작동하는지 테스트합니다."""
    
    print("🧪 Evidence 추출 모델별 경로 테스트")
    print("=" * 60)
    
    # 테스트할 모델과 도메인
    test_cases = [
        ("deepseek-r1:7b", "medical"),
        ("llama2:7b", "medical"),
        ("mistral:7b", "medical"),
        ("gemma:7b", "medical"),
    ]
    
    for model_key, domain in test_cases:
        print(f"\n📋 테스트 케이스: {model_key} - {domain}")
        print("-" * 40)
        
        # 1. load_origin_prompts 테스트
        print(f"1️⃣ load_origin_prompts 테스트:")
        prompts = load_origin_prompts(domain, model_key)
        print(f"   결과: {len(prompts)}개 프롬프트 로드됨")
        
        if prompts:
            print(f"   첫 번째 프롬프트: {prompts[0].get('prompt', 'N/A')[:50]}...")
        
        # 2. get_available_files 테스트
        print(f"2️⃣ get_available_files 테스트:")
        files = get_available_files(domain, model_key)
        print(f"   결과: {len(files)}개 파일 발견")
        
        for file_info in files[:3]:  # 처음 3개만 표시
            print(f"   - {file_info['name']}: {file_info['prompt_count']}개 프롬프트")
        
        # 3. load_selected_files 테스트 (파일이 있는 경우)
        if files:
            print(f"3️⃣ load_selected_files 테스트:")
            selected_files = [files[0]['name']]  # 첫 번째 파일 선택
            selected_prompts = load_selected_files(domain, selected_files, model_key)
            print(f"   결과: {len(selected_prompts)}개 프롬프트 로드됨")
        
        print()

def test_domain_discovery():
    """도메인 발견 로직을 테스트합니다."""
    
    print("\n🔍 도메인 발견 로직 테스트")
    print("=" * 60)
    
    # 테스트할 모델들
    test_models = ["deepseek-r1:7b", "llama2:7b", "mistral:7b", "gemma:7b"]
    
    for model_key in test_models:
        print(f"\n📋 모델: {model_key}")
        print("-" * 30)
        
        # 모델별 도메인 디렉토리 확인
        model_origin_dir = Path(f"dataset/origin/{model_key.replace(':', '_')}")
        if not model_origin_dir.exists():
            model_origin_dir = Path(f"dataset/origin/{model_key}")
        
        if model_origin_dir.exists():
            domains = [d.name for d in model_origin_dir.iterdir() if d.is_dir()]
            print(f"   발견된 도메인: {domains}")
            
            # 각 도메인별 프롬프트 수 확인
            for domain in domains:
                prompts = load_origin_prompts(domain, model_key)
                print(f"   - {domain}: {len(prompts)}개 프롬프트")
        else:
            print(f"   ❌ 모델별 디렉토리가 존재하지 않음: {model_origin_dir}")

def test_legacy_paths():
    """기존 경로(모델별 디렉토리 없음)도 작동하는지 테스트합니다."""
    
    print("\n🔄 기존 경로 호환성 테스트")
    print("=" * 60)
    
    # 기존 방식으로 도메인 확인
    origin_dir = Path("dataset/origin")
    if origin_dir.exists():
        legacy_domains = [d.name for d in origin_dir.iterdir() if d.is_dir() and not ':' in d.name]
        print(f"기존 방식 도메인: {legacy_domains}")
        
        for domain in legacy_domains[:2]:  # 처음 2개만 테스트
            print(f"\n📋 도메인: {domain}")
            prompts = load_origin_prompts(domain)  # model_key 없이 호출
            print(f"   결과: {len(prompts)}개 프롬프트 로드됨")

if __name__ == "__main__":
    test_model_paths()
    test_domain_discovery()
    test_legacy_paths()
    
    print("\n✅ 테스트 완료!") 