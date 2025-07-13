#!/usr/bin/env python3
"""
각 프롬프트마다 고유한 evidence가 추출되는지 테스트하는 스크립트
"""

import sys
import json
from pathlib import Path

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

from evidence_extractor import extract_evidence_tokens

def test_unique_evidence():
    """각 프롬프트마다 고유한 evidence가 추출되는지 테스트합니다."""
    
    print("🧪 고유 Evidence 추출 테스트")
    print("=" * 60)
    
    # 테스트할 프롬프트들 (의도적으로 다른 내용)
    test_prompts = [
        "What are the symptoms of a heart attack?",
        "How does diabetes affect the cardiovascular system?",
        "What are the risk factors for developing cancer?",
        "Explain the mechanism of action of antibiotics.",
        "What are the common causes of high blood pressure?"
    ]
    
    model_key = "mistral:7b"
    domain = "medical"
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📋 테스트 {i}: {prompt}")
        print("-" * 50)
        
        # Evidence 추출
        evidence_indices, evidence_tokens = extract_evidence_tokens(prompt, model_key, domain)
        
        result = {
            "prompt": prompt,
            "evidence_tokens": evidence_tokens,
            "evidence_indices": evidence_indices
        }
        results.append(result)
        
        print(f"결과:")
        print(f"  - Evidence 토큰: {evidence_tokens}")
        print(f"  - Evidence 인덱스: {evidence_indices}")
        print()
    
    # 결과 분석
    print("📊 결과 분석")
    print("=" * 60)
    
    all_tokens = []
    for result in results:
        all_tokens.extend(result["evidence_tokens"])
    
    print(f"전체 추출된 토큰: {all_tokens}")
    print(f"고유 토큰: {list(set(all_tokens))}")
    print(f"토큰 중복률: {len(all_tokens) - len(set(all_tokens))}/{len(all_tokens)}")
    
    # 각 프롬프트별 고유성 확인
    print(f"\n각 프롬프트별 고유성:")
    for i, result in enumerate(results, 1):
        prompt_tokens = result["evidence_tokens"]
        unique_tokens = list(set(prompt_tokens))
        print(f"  프롬프트 {i}: {len(prompt_tokens)}개 토큰, {len(unique_tokens)}개 고유")
        print(f"    토큰: {prompt_tokens}")
    
    # 결과 저장
    with open("evidence_uniqueness_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 결과가 'evidence_uniqueness_test.json'에 저장되었습니다.")

if __name__ == "__main__":
    test_unique_evidence()
    print("\n✅ 테스트 완료!") 