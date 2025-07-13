#!/usr/bin/env python3
"""
Gemma 모델의 빠른 evidence 추출 테스트
"""

import sys
import json
import time
from pathlib import Path

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

from evidence_extractor import extract_evidence_tokens

def test_gemma_fast():
    """Gemma 모델의 빠른 evidence 추출을 테스트합니다."""
    
    print("🚀 Gemma 빠른 Evidence 추출 테스트")
    print("=" * 50)
    
    # 간단한 테스트 프롬프트
    test_prompt = "How does inflation affect market equilibrium?"
    model_key = "gemma:7b"
    domain = "economy"
    
    print(f"📝 프롬프트: {test_prompt}")
    print(f"🤖 모델: {model_key}")
    print(f"🏢 도메인: {domain}")
    print()
    
    # 타임아웃 설정
    start_time = time.time()
    
    try:
        print("🔍 Evidence 추출 시작...")
        evidence_indices, evidence_tokens = extract_evidence_tokens(test_prompt, model_key, domain)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if evidence_tokens:
            print(f"✅ 성공! ({duration:.2f}초)")
            print(f"📊 추출된 토큰: {evidence_tokens}")
            print(f"📍 토큰 인덱스: {evidence_indices}")
        else:
            print(f"❌ 실패! ({duration:.2f}초)")
            print("Evidence 토큰을 추출할 수 없습니다.")
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ 오류 발생! ({duration:.2f}초)")
        print(f"오류 내용: {str(e)}")

if __name__ == "__main__":
    test_gemma_fast() 