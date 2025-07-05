import json
import os
from collections import defaultdict

def extract_prompts_from_jsonl(file_path):
    """JSONL 파일에서 프롬프트만 추출"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                prompts.append({
                    "prompt": data["prompt"],
                    "domain": data["domain"],
                    "model": data["model"],
                    "index": data["index"]
                })
            except json.JSONDecodeError:
                continue
    return prompts

def main():
    # 각 도메인별로 처리할 파일들
    domain_files = {
        "economy": "dataset/gemma:7b/economy/gemma:7b_1000prompts_20250623_074130.jsonl",
        "legal": "dataset/gemma:7b/legal/gemma:7b_1000prompts_20250623_042404.jsonl", 
        "medical": "dataset/gemma:7b/medical/gemma:7b_1000prompts_20250623_023509.jsonl",
        "technical": "dataset/gemma:7b/technical/gemma:7b_1000prompts_20250623_061457.jsonl"
    }
    
    all_prompts = []
    
    for domain, file_path in domain_files.items():
        if os.path.exists(file_path):
            print(f"처리 중: {domain} - {file_path}")
            prompts = extract_prompts_from_jsonl(file_path)
            all_prompts.extend(prompts)
            print(f"  - {len(prompts)}개 프롬프트 추출됨")
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    # 결과를 JSON 파일로 저장
    output_file = "extracted_prompts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"\n총 {len(all_prompts)}개 프롬프트가 {output_file}에 저장되었습니다.")
    
    # 도메인별 통계 출력
    domain_stats = defaultdict(int)
    for prompt in all_prompts:
        domain_stats[prompt["domain"]] += 1
    
    print("\n도메인별 프롬프트 개수:")
    for domain, count in domain_stats.items():
        print(f"  {domain}: {count}개")

if __name__ == "__main__":
    main() 