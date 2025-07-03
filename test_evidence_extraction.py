#!/usr/bin/env python3
"""
실제 Evidence 추출 기능 테스트 스크립트
"""

from transformers import AutoTokenizer
import requests
import json
import re
from typing import List, Dict, Any

# Ollama API 설정
OLLAMA_API_BASE = "http://localhost:11434"

def get_model_response(model_name: str, prompt: str) -> str:
    """Ollama 모델에서 응답을 받아옵니다."""
    try:
        response = requests.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise Exception(f"API 요청 실패: {response.status_code}")
    except Exception as e:
        raise Exception(f"모델 응답 오류: {str(e)}")

def create_evidence_query(word_list: List[str], prompt: str, domain: str) -> str:
    """Evidence 추출을 위한 프롬프트를 생성합니다."""
    
    evidence_query = f"""Extract key evidence words from the following {domain} domain question.

Question: "{prompt}"

Provided word list:
{', '.join(word_list)}

Please respond in the following JSON format:
{{
    "evidence_words": ["word1", "word2", "word3"],
    "explanation": "Reason for selecting these words"
}}

Guidelines:
1. Select words that are crucial for understanding the core of the question
2. Prioritize domain-specific technical terms
3. There is no limit on the number of words you can select
4. Only select words from the provided word list"""

    return evidence_query

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """응답에서 JSON을 추출합니다."""
    try:
        # JSON 블록 찾기
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            # JSON이 없으면 수동으로 파싱 시도
            evidence_words = []
            explanation = ""
            
            # evidence_words 추출
            words_match = re.search(r'"evidence_words":\s*\[(.*?)\]', response, re.DOTALL)
            if words_match:
                words_str = words_match.group(1)
                evidence_words = [word.strip().strip('"\'') for word in words_str.split(',') if word.strip()]
            
            # explanation 추출
            exp_match = re.search(r'"explanation":\s*"([^"]*)"', response)
            if exp_match:
                explanation = exp_match.group(1)
            
            return {
                "evidence_words": evidence_words,
                "explanation": explanation
            }
    except Exception as e:
        print(f"JSON 파싱 오류: {str(e)}")
        return {"evidence_words": [], "explanation": "파싱 실패"}

def validate_evidence(result: Dict[str, Any], words: List[str]) -> Dict[str, Any]:
    """추출된 evidence를 검증합니다."""
    evidence_words = result.get("evidence_words", [])
    explanation = result.get("explanation", "")
    
    # 제공된 단어 목록에 있는지 확인
    valid_words = []
    invalid_words = []
    
    for word in evidence_words:
        if word in words:
            valid_words.append(word)
        else:
            invalid_words.append(word)
    
    return {
        "evidence_words": valid_words,
        "invalid_words": invalid_words,
        "explanation": explanation,
        "is_valid": len(invalid_words) == 0
    }

def extract_evidence_with_deepseek(prompt: str, model_name: str = "deepseek-r1:7b", domain: str = "Medical") -> Dict[str, Any]:
    """DeepSeek 모델을 사용하여 evidence를 추출합니다."""
    
    print(f"🧠 Evidence 추출 시작...")
    print(f"📝 프롬프트: {prompt}")
    print(f"🤖 모델: {model_name}")
    print(f"🏥 도메인: {domain}")
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base", trust_remote_code=True)
        
        # 텍스트를 단어로 분리
        words = prompt.split()
        print(f"📚 단어 목록 ({len(words)}개): {words}")
        
        # Evidence 추출 쿼리 생성
        evidence_query = create_evidence_query(words, prompt, domain)
        print(f"\n🔍 Evidence 추출 쿼리:")
        print(evidence_query)
        
        # 모델에 쿼리 전송
        print(f"\n📤 모델에 쿼리 전송 중...")
        response = get_model_response(model_name, evidence_query)
        print(f"📥 모델 응답:")
        print(response)
        
        # JSON 추출
        result = extract_json_from_response(response)
        print(f"\n🔍 추출된 JSON: {result}")
        
        # 검증
        validated_result = validate_evidence(result, words)
        print(f"\n✅ 검증 결과: {validated_result}")
        
        return {
            "prompt": prompt,
            "model": model_name,
            "domain": domain,
            "words": words,
            "evidence_words": validated_result["evidence_words"],
            "invalid_words": validated_result["invalid_words"],
            "explanation": validated_result["explanation"],
            "is_valid": validated_result["is_valid"],
            "raw_response": response
        }
        
    except Exception as e:
        print(f"❌ Evidence 추출 실패: {str(e)}")
        return {
            "error": str(e),
            "prompt": prompt,
            "model": model_name,
            "domain": domain
        }

def main():
    """메인 테스트 함수"""
    
    print("🚀 DeepSeek Evidence 추출 테스트")
    print("="*60)
    
    # 테스트 프롬프트들
    test_cases = [
        {
            "prompt": "What are the key clinical features associated with a diagnosis of atrial fibrillation in patients presenting with stroke?",
            "domain": "Medical"
        },
        {
            "prompt": "Describe the primary ethical considerations associated with the use of artificial intelligence in patient diagnosis and treatment.",
            "domain": "Medical"
        },
        {
            "prompt": "List the most common side effects associated with long-term use of Metformin for Type 2 Diabetes management.",
            "domain": "Medical"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 테스트 케이스 {i}")
        print(f"{'='*60}")
        
        result = extract_evidence_with_deepseek(
            prompt=test_case["prompt"],
            domain=test_case["domain"]
        )
        
        results.append(result)
        
        # 결과 요약
        if "error" not in result:
            print(f"\n📋 결과 요약:")
            print(f"  - 추출된 evidence 단어: {result['evidence_words']}")
            print(f"  - 유효성: {'✅ 유효' if result['is_valid'] else '❌ 무효'}")
            if result['invalid_words']:
                print(f"  - 무효한 단어: {result['invalid_words']}")
            print(f"  - 설명: {result['explanation']}")
        else:
            print(f"❌ 오류: {result['error']}")
    
    # 전체 결과 저장
    with open("evidence_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"📊 전체 테스트 완료!")
    print(f"📁 결과가 'evidence_extraction_results.json'에 저장되었습니다.")
    
    # 성공률 계산
    success_count = sum(1 for r in results if "error" not in r and r.get("is_valid", False))
    total_count = len(results)
    
    print(f"✅ 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    main() 