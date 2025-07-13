#!/usr/bin/env python3
"""
Evidence 추출 과정을 디버깅하는 스크립트
"""

import sys
import json
from pathlib import Path

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

from evidence_extractor import extract_evidence_tokens, call_ollama_api

def debug_evidence_extraction():
    """Evidence 추출 과정을 디버깅합니다."""
    
    print("🔍 Evidence 추출 디버깅")
    print("=" * 60)
    
    # 테스트할 프롬프트들
    test_prompts = [
        "What are the symptoms of a heart attack?",
        "What are the common causes of high blood pressure?",
        "How does diabetes affect the cardiovascular system?",
        "What are the risk factors for developing cancer?",
        "Explain the mechanism of action of antibiotics."
    ]
    
    model_key = "mistral:7b"
    domain = "medical"
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📋 테스트 {i}: {prompt}")
        print("-" * 50)
        
        # Evidence 추출
        evidence_indices, evidence_tokens = extract_evidence_tokens(prompt, model_key, domain)
        
        print(f"결과:")
        print(f"  - Evidence 토큰: {evidence_tokens}")
        print(f"  - Evidence 인덱스: {evidence_indices}")
        print()

def debug_ollama_response():
    """Ollama API 응답을 디버깅합니다."""
    
    print("🔍 Ollama API 응답 디버깅")
    print("=" * 60)
    
    # 테스트할 프롬프트
    prompt = "What are the symptoms of a heart attack?"
    model_key = "mistral:7b"
    domain = "medical"
    
    # Evidence 추출을 위한 프롬프트 구성
    evidence_prompt = f"""
You are an English-only evidence extraction system. Your task is to extract English tokens from the given prompt.

DOMAIN: {domain}
INSTRUCTION: Find important tokens related to the domain.

INPUT PROMPT: "{prompt}"

CRITICAL RULES:
1. You MUST respond ONLY in English
2. You MUST extract ONLY English words that exist in the input prompt
3. You MUST return ONLY a JSON array format
4. You MUST NOT translate words
5. You MUST NOT add words that are not in the prompt
6. You MUST NOT respond in Korean or any other language
7. You MUST focus on domain-specific medical/technical terms

EXAMPLES:
- For medical domain: ["clinical", "findings", "diagnosis", "viral", "encephalitis", "adults"]
- For legal domain: ["legal", "contract", "liability", "jurisdiction"]
- For technical domain: ["algorithm", "implementation", "optimization"]

RESPONSE FORMAT (JSON array only):
["word1", "word2", "word3"]
"""
    
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

def debug_token_extraction():
    """토큰 추출 과정을 디버깅합니다."""
    
    print("🔍 토큰 추출 디버깅")
    print("=" * 60)
    
    # 테스트할 응답들
    test_responses = [
        '["symptoms", "heart", "attack"]',
        '["common", "causes", "high", "blood", "pressure"]',
        '["diabetes", "affects", "cardiovascular", "system"]',
        '["risk", "factors", "developing", "cancer"]',
        '["mechanism", "action", "antibiotics"]'
    ]
    
    from evidence_extractor import extract_tokens_from_response
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n📋 테스트 {i}: {response}")
        print("-" * 30)
        
        tokens = extract_tokens_from_response(response)
        print(f"추출된 토큰: {tokens}")
        print()

if __name__ == "__main__":
    debug_evidence_extraction()
    debug_ollama_response()
    debug_token_extraction()
    
    print("✅ 디버깅 완료!") 