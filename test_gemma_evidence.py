#!/usr/bin/env python3
"""
Gemma 모델의 evidence 추출을 테스트하는 스크립트
"""

import sys
import json
from pathlib import Path

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

from evidence_extractor import extract_evidence_tokens, call_ollama_api

def test_gemma_evidence():
    """Gemma 모델의 evidence 추출을 테스트합니다."""
    
    print("🔍 Gemma Evidence 추출 테스트")
    print("=" * 60)
    
    # 테스트할 프롬프트들
    test_prompts = [
        "How does inflation affect the equilibrium point of supply and demand in a competitive market?",
        "What are the symptoms of a heart attack?",
        "How does government regulation influence market efficiency?",
        "What are the risk factors for developing diabetes?"
    ]
    
    model_key = "gemma:7b"
    domain = "economy"
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📋 테스트 {i}: {prompt}")
        print("-" * 50)
        
        # Evidence 추출
        evidence_indices, evidence_tokens = extract_evidence_tokens(prompt, model_key, domain)
        
        print(f"결과:")
        print(f"  - Evidence 토큰: {evidence_tokens}")
        print(f"  - Evidence 인덱스: {evidence_indices}")
        print(f"  - 토큰 개수: {len(evidence_tokens)}")
        print()

def test_gemma_raw_response():
    """Gemma 모델의 원시 응답을 테스트합니다."""
    
    print("🔍 Gemma 원시 응답 테스트")
    print("=" * 60)
    
    # 테스트할 프롬프트
    prompt = "How does inflation affect the equilibrium point of supply and demand in a competitive market?"
    model_key = "gemma:7b"
    domain = "economy"
    
    # Evidence 추출을 위한 프롬프트 구성
    evidence_prompt = f"""<start_of_turn>user
Extract ONLY English words from this prompt that are relevant to the {domain} domain.

INPUT: "{prompt}"

RULES:
- Extract ONLY single words from the input
- Return ONLY JSON array format
- NO explanations or additional text
- NO compound phrases - split "heart attack" into ["heart", "attack"]
- Focus on domain-specific terms

EXAMPLES:
- Medical: ["symptoms", "heart", "attack", "diagnosis"]
- Legal: ["contract", "liability", "jurisdiction"]
- Technical: ["algorithm", "implementation", "optimization"]

RESPONSE (JSON array only):
["word1", "word2", "word3"]<end_of_turn>
<start_of_turn>model
["word1", "word2", "word3"]<end_of_turn>"""
    
    print(f"📤 Ollama API 호출:")
    print(f"모델: {model_key}")
    print(f"프롬프트: {prompt}")
    print(f"도메인: {domain}")
    print()
    
    # Ollama API 호출
    response_text = call_ollama_api(model_key, evidence_prompt)
    
    print(f"📥 Ollama API 응답:")
    print(f"응답 길이: {len(response_text) if response_text else 0} 문자")
    print(f"응답 내용:")
    print(response_text if response_text else "No response")
    print()

if __name__ == "__main__":
    print("🚀 Gemma Evidence 추출 테스트 시작")
    print()
    
    # 원시 응답 테스트
    test_gemma_raw_response()
    
    print("\n" + "="*60)
    print()
    
    # Evidence 추출 테스트
    test_gemma_evidence()
    
    print("✅ 테스트 완료") 