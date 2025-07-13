#!/usr/bin/env python3
"""
Gemma 모델의 evidence 추출 결과 저장 문제를 디버깅하는 스크립트
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# evidence_extractor 모듈의 함수들을 가져오기 위해 경로 추가
sys.path.append('tabs')

from evidence_extractor import extract_evidence_tokens, save_domain_data

def test_gemma_save():
    """Gemma 모델의 evidence 추출 결과 저장을 테스트합니다."""
    
    print("🔍 Gemma Evidence 저장 테스트")
    print("=" * 60)
    
    # 테스트할 프롬프트들
    test_prompts = [
        "How does inflation affect the equilibrium point of supply and demand in a competitive market?",
        "What are the effects of government regulation on market efficiency?",
        "How do interest rates influence economic growth and investment decisions?"
    ]
    
    model_key = "gemma:7b"
    domain = "economy"
    
    # Evidence 추출 테스트
    extracted_data = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n📝 테스트 {i+1}: {prompt[:50]}...")
        
        try:
            # Evidence 추출
            evidence_indices, evidence_tokens = extract_evidence_tokens(prompt, model_key, domain)
            
            if evidence_tokens:
                # 데이터 구조 생성
                data_item = {
                    "prompt": prompt,
                    "domain": domain,
                    "model": model_key,
                    "index": i,
                    "evidence_indices": evidence_indices,
                    "evidence_tokens": evidence_tokens,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                extracted_data.append(data_item)
                print(f"   ✅ Evidence 추출 성공: {len(evidence_tokens)}개 토큰")
                print(f"   추출된 토큰들: {evidence_tokens}")
            else:
                print(f"   ❌ Evidence 추출 실패")
                
        except Exception as e:
            print(f"   ❌ 오류 발생: {str(e)}")
    
    print(f"\n📊 총 추출된 데이터: {len(extracted_data)}개")
    
    if extracted_data:
        # 저장 테스트
        print(f"\n💾 저장 테스트 시작...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            output_path, saved_count = save_domain_data(domain, extracted_data, model_key, timestamp)
            
            if output_path and saved_count > 0:
                print(f"✅ 저장 성공!")
                print(f"   저장된 파일: {output_path}")
                print(f"   저장된 항목 수: {saved_count}")
                
                # 파일 내용 확인
                if output_path.exists():
                    print(f"\n📄 저장된 파일 내용 확인:")
                    with open(output_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"   파일 크기: {output_path.stat().st_size} bytes")
                        print(f"   라인 수: {len(lines)}")
                        
                        # 첫 번째 라인 확인
                        if lines:
                            first_line = json.loads(lines[0])
                            print(f"   첫 번째 항목 키: {list(first_line.keys())}")
                            print(f"   Evidence tokens: {first_line.get('evidence_tokens', [])}")
                else:
                    print(f"❌ 파일이 생성되지 않음: {output_path}")
            else:
                print(f"❌ 저장 실패: output_path={output_path}, saved_count={saved_count}")
                
        except Exception as e:
            print(f"❌ 저장 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ 저장할 데이터가 없습니다.")

if __name__ == "__main__":
    test_gemma_save() 