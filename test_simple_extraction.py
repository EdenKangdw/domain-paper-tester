#!/usr/bin/env python3
"""
PyTorch 오류 없이 간단한 evidence 추출 테스트
"""

import sys
import json
import time
from pathlib import Path

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

def simple_evidence_extraction():
    """PyTorch 없이 간단한 evidence 추출을 테스트합니다."""
    
    print("🚀 간단한 Evidence 추출 테스트")
    print("=" * 50)
    
    # 간단한 테스트 프롬프트
    test_prompts = [
        "How does inflation affect market equilibrium?",
        "What are the effects of government regulation?",
        "How do interest rates influence economic growth?"
    ]
    
    model_key = "gemma:7b"
    domain = "economy"
    
    # Ollama API 직접 호출
    import requests
    
    OLLAMA_API_BASE = "http://localhost:11434"
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 테스트 {i+1}: {prompt[:50]}...")
        
        try:
            # 간단한 프롬프트
            evidence_prompt = f"""<start_of_turn>user
Extract key words from: "{prompt}"

Domain: {domain}
Format: JSON array only
Example: ["word1", "word2", "word3"]<end_of_turn>
<start_of_turn>model
["word1", "word2", "word3"]<end_of_turn>"""
            
            # API 호출
            start_time = time.time()
            response = requests.post(
                f"{OLLAMA_API_BASE}/api/generate",
                json={
                    "model": model_key,
                    "prompt": evidence_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.05,
                        "top_p": 0.8,
                        "num_predict": 50,
                        "repeat_penalty": 1.0,
                        "top_k": 3
                    }
                },
                timeout=10
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # 간단한 파싱
                import re
                quoted_tokens = re.findall(r'["\']([^"\']+)["\']', response_text)
                if quoted_tokens:
                    tokens = list(set([token.strip() for token in quoted_tokens if token.strip()]))
                    print(f"   ✅ 성공! ({duration:.2f}초)")
                    print(f"   📊 추출된 토큰: {tokens}")
                else:
                    print(f"   ⚠️ 토큰 추출 실패 ({duration:.2f}초)")
                    print(f"   응답: {response_text}")
            else:
                print(f"   ❌ API 오류 ({duration:.2f}초): {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    simple_evidence_extraction() 