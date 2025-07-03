import json
import os
import shutil
from collections import defaultdict

def split_prompts_by_domain():
    """extracted_prompts.json을 도메인별로 나누어서 저장"""
    
    # extracted_prompts.json 파일 읽기
    with open("extracted_prompts.json", 'r', encoding='utf-8') as f:
        all_prompts = json.load(f)
    
    print(f"총 {len(all_prompts)}개 프롬프트 로드됨")
    
    # 도메인별로 분류
    domain_prompts = defaultdict(list)
    for prompt in all_prompts:
        domain = prompt["domain"]
        domain_prompts[domain].append(prompt)
    
    # 각 도메인별로 JSON 파일 생성
    output_dir = "dataset/origin"
    os.makedirs(output_dir, exist_ok=True)
    
    for domain, prompts in domain_prompts.items():
        # 도메인별 폴더 생성
        domain_dir = os.path.join(output_dir, domain.lower())
        os.makedirs(domain_dir, exist_ok=True)
        
        # JSON 파일 저장
        output_file = os.path.join(domain_dir, f"{domain.lower()}_prompts.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        print(f"{domain}: {len(prompts)}개 프롬프트 -> {output_file}")
    
    # extracted_prompts.json을 origin 폴더로 복사
    shutil.copy("extracted_prompts.json", os.path.join(output_dir, "all_prompts.json"))
    print(f"\nextracted_prompts.json이 {output_dir}/all_prompts.json으로 복사됨")
    
    print(f"\n총 {len(domain_prompts)}개 도메인으로 분리 완료:")
    for domain, prompts in domain_prompts.items():
        print(f"  {domain}: {len(prompts)}개")

if __name__ == "__main__":
    split_prompts_by_domain() 