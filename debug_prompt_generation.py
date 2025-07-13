#!/usr/bin/env python3
"""
프롬프트 생성 디버깅 스크립트
"""

import requests
import json

# Ollama API 설정
OLLAMA_API_BASE = "http://localhost:11434"

def test_model_response(model_name, prompt):
    """모델 응답을 테스트합니다."""
    try:
        print(f"Testing model: {model_name}")
        print(f"Prompt: {prompt}")
        
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 200
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            
            print(f"Response: {response_text}")
            print(f"Response length: {len(response_text)}")
            
            # 응답 검증
            if not response_text:
                print("❌ Empty response")
                return False
            elif response_text.lower().startswith(('please enter', 'error', 'failed', 'i cannot', 'i am unable')):
                print("❌ Invalid response")
                return False
            elif len(response_text) < 5:
                print("❌ Response too short")
                return False
            else:
                print("✅ Valid response")
                return True
        else:
            print(f"❌ API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """메인 함수"""
    print("🔍 프롬프트 생성 디버깅 시작")
    
    # 테스트할 모델들 (실제 모델 이름)
    models = ["llama2:7b", "deepseek-r1:7b"]
    
    # 도메인별 테스트 프롬프트 (딥시크 모델용 구체적 버전)
    domain_prompts = {
        "Medical": "Write only one clear medical question about symptoms, treatment, or health. Do not explain your reasoning. Do not include <think> or any thoughts. Only output the question sentence itself. Example: What are the symptoms of diabetes?",
        "Legal": "Write only one clear legal question about contracts, rights, or procedures. Do not explain your reasoning. Do not include <think> or any thoughts. Only output the question sentence itself. Example: What are the requirements for a valid contract?",
        "Technical": "Write only one clear technical question about computers or technology. Do not explain your reasoning. Do not include <think> or any thoughts. Only output the question sentence itself. Example: How does machine learning work?",
        "Economy": "Write only one clear economic question about markets or finance. Do not explain your reasoning. Do not include <think> or any thoughts. Only output the question sentence itself. Example: How do interest rates affect the economy?"
    }
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Testing model: {model}")
        print(f"{'='*50}")
        
        for domain, prompt in domain_prompts.items():
            print(f"\n--- {domain} ---")
            success = test_model_response(model, prompt)
            if not success:
                print(f"⚠️ {model} failed for {domain}")
            print()

if __name__ == "__main__":
    main() 