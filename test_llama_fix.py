#!/usr/bin/env python3
"""
라마 모델 개선된 프롬프트 테스트
"""

import requests
import json
import re
import ast

# 테스트용 프롬프트
prompt = 'What are the symptoms of a heart attack?'
domain = 'medical'

evidence_prompt = f'''<s>[INST] Extract ONLY English words from this prompt that are relevant to the {domain} domain.

INPUT: "{prompt}"

RULES:
- Extract ONLY single words from the input
- NO explanations or text
- NO compound phrases - split "heart attack" into ["heart", "attack"]
- Return ONLY JSON array

EXAMPLES:
- "heart attack symptoms" → ["heart", "attack", "symptoms"]
- "processing power" → ["processing", "power"]

RESPONSE (JSON only):
["word1", "word2", "word3"] [/INST]'''

print('🔍 Llama2:7b 모델 개선된 프롬프트 테스트')
print('=' * 60)
print(f'프롬프트: {prompt}')
print(f'도메인: {domain}')
print()

try:
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama2:7b',
            'prompt': evidence_prompt,
            'stream': False,
            'options': {
                'temperature': 0.3,
                'top_p': 0.8,
                'num_predict': 100,
                'repeat_penalty': 1.2,
                'top_k': 10
            }
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        response_text = result.get('response', '').strip()
        print(f'✅ API 호출 성공: {len(response_text)} 문자')
        print(f'응답 내용:')
        print(response_text)
        
        # 토큰 추출 테스트
        # 라마 모델 응답 정리
        cleaned_response = response_text.strip()
        if any(keyword in cleaned_response for keyword in ["Sure!", "Here are", "Here is", "The", "This", "I'll", "Let me"]):
            json_match = re.search(r'\[[^\]]+\]', cleaned_response)
            if json_match:
                cleaned_response = json_match.group()
                print(f'\n🔧 응답 정리 후: {cleaned_response}')
                print(f'원본: {response_text[:100]}...')
        
        # JSON 배열 추출
        list_match = re.search(r'\[[^\]]+\]', cleaned_response)
        if list_match:
            try:
                evidence_tokens = ast.literal_eval(list_match.group())
                if isinstance(evidence_tokens, list):
                    result = [str(token).strip() for token in evidence_tokens if token]
                    print(f'\n✅ 추출된 토큰: {result}')
                    
                    # 복합어 분리 테스트
                    english_tokens = []
                    for token in result:
                        token_clean = token.lower().strip()
                        if ' ' in token_clean:
                            print(f'🔧 복합어 분리: "{token_clean}" -> {token_clean.split()}')
                            english_tokens.extend(token_clean.split())
                        else:
                            english_tokens.append(token_clean)
                    
                    print(f'\n🔍 최종 토큰: {english_tokens}')
                    
                    # 원본 프롬프트에서 인덱스 찾기 테스트
                    print(f'\n🔍 원본 프롬프트에서 인덱스 찾기:')
                    for token in english_tokens:
                        index = prompt.lower().find(token.lower())
                        if index != -1:
                            print(f'   ✅ "{token}" 발견: 인덱스 {index}')
                        else:
                            print(f'   ⚠️ "{token}" 찾을 수 없음')
                else:
                    print('❌ JSON 배열이 아님')
            except Exception as e:
                print(f'❌ JSON 파싱 실패: {str(e)}')
        else:
            print('❌ JSON 배열을 찾을 수 없음')
            
    else:
        print(f'❌ API 호출 실패: {response.status_code}')
        print(response.text)
        
except Exception as e:
    print(f'❌ 오류: {str(e)}') 