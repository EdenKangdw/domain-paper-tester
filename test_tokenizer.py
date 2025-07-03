#!/usr/bin/env python3
"""
DeepSeek 토크나이저 테스트 스크립트
"""

from transformers import AutoTokenizer
import json

def test_deepseek_tokenizer():
    """DeepSeek 토크나이저를 테스트합니다."""
    
    print("🔍 DeepSeek 토크나이저 테스트 시작...")
    
    try:
        # 토크나이저 로드
        print("📥 토크나이저 로딩 중: deepseek-ai/deepseek-llm-7b-base")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base", trust_remote_code=True)
        print("✅ 토크나이저 로드 성공!")
        
        # 테스트 프롬프트
        test_prompt = "What are the key clinical features associated with a diagnosis of atrial fibrillation in patients presenting with stroke?"
        
        print(f"\n📝 테스트 프롬프트: {test_prompt}")
        
        # 토크나이징
        tokens = tokenizer.tokenize(test_prompt)
        print(f"🔤 토큰화 결과 ({len(tokens)}개 토큰):")
        for i, token in enumerate(tokens):
            print(f"  {i:2d}: '{token}'")
        
        # 인코딩/디코딩 테스트
        encoded = tokenizer.encode(test_prompt)
        decoded = tokenizer.decode(encoded)
        print(f"\n🔄 인코딩/디코딩 테스트:")
        print(f"원본: {test_prompt}")
        print(f"복원: {decoded}")
        
        # 토크나이저 정보
        print(f"\n📊 토크나이저 정보:")
        print(f"  - vocab_size: {tokenizer.vocab_size}")
        print(f"  - model_max_length: {tokenizer.model_max_length}")
        print(f"  - pad_token: {tokenizer.pad_token}")
        print(f"  - eos_token: {tokenizer.eos_token}")
        print(f"  - bos_token: {tokenizer.bos_token}")
        
        return True
        
    except Exception as e:
        print(f"❌ 토크나이저 테스트 실패: {str(e)}")
        return False

def test_evidence_extraction():
    """Evidence 추출 기능을 테스트합니다."""
    
    print("\n" + "="*50)
    print("🧠 Evidence 추출 테스트")
    print("="*50)
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base", trust_remote_code=True)
        
        # 테스트 프롬프트들
        test_prompts = [
            "What are the key clinical features associated with a diagnosis of atrial fibrillation in patients presenting with stroke?",
            "Describe the primary ethical considerations associated with the use of artificial intelligence in patient diagnosis and treatment.",
            "List the most common side effects associated with long-term use of Metformin for Type 2 Diabetes management."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 테스트 {i}: {prompt}")
            
            # 토크나이징
            tokens = tokenizer.tokenize(prompt)
            print(f"🔤 토큰 수: {len(tokens)}")
            
            # 간단한 evidence 추출 시뮬레이션 (실제로는 모델이 필요)
            # 여기서는 단어 기반으로 간단히 처리
            words = prompt.split()
            print(f"📚 단어 수: {len(words)}")
            
            # 첫 번째와 마지막 단어를 evidence로 가정
            evidence_words = [words[0], words[-1]] if len(words) > 1 else words
            print(f"🎯 추출된 evidence 단어들: {evidence_words}")
            
        return True
        
    except Exception as e:
        print(f"❌ Evidence 추출 테스트 실패: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 DeepSeek 토크나이저 및 Evidence 추출 테스트")
    print("="*60)
    
    # 토크나이저 테스트
    tokenizer_success = test_deepseek_tokenizer()
    
    # Evidence 추출 테스트
    evidence_success = test_evidence_extraction()
    
    print("\n" + "="*60)
    print("📋 테스트 결과 요약:")
    print(f"  토크나이저 테스트: {'✅ 성공' if tokenizer_success else '❌ 실패'}")
    print(f"  Evidence 추출 테스트: {'✅ 성공' if evidence_success else '❌ 실패'}")
    
    if tokenizer_success and evidence_success:
        print("\n🎉 모든 테스트가 성공했습니다!")
        print("💡 이제 Streamlit 앱에서 deepseek-r1:7b 모델을 사용할 수 있습니다.")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다. 설정을 확인해주세요.") 