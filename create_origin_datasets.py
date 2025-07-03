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

def save_domain_prompts(prompts, domain, output_dir):
    """도메인별로 프롬프트를 JSON 파일로 저장"""
    domain_prompts = [p for p in prompts if p["domain"].lower() == domain.lower()]
    
    # 10000개로 제한
    domain_prompts = domain_prompts[:10000]
    
    # 도메인별 폴더 생성
    domain_dir = os.path.join(output_dir, domain.lower())
    os.makedirs(domain_dir, exist_ok=True)
    
    output_file = os.path.join(domain_dir, f"{domain.lower()}_prompts.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(domain_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"{domain}: {len(domain_prompts)}개 프롬프트 저장됨 -> {output_file}")
    return len(domain_prompts)

def main():
    # mistral 데이터 파일들
    mistral_files = [
        "dataset/mistral_7b/general/mistral:7b_1000prompts_20250617_083304.jsonl",
        "dataset/mistral_7b/legal/mistral:7b_1000prompts_20250617_050705.jsonl",
        "dataset/mistral_7b/medical/mistral:7b_1000prompts_20250617_021449.jsonl",
        "dataset/mistral_7b/technical/mistral:7b_1000prompts_20250617_063930.jsonl"
    ]
    
    output_dir = "dataset/origin"
    
    all_prompts = []
    
    # 모든 mistral 파일에서 프롬프트 추출
    for file_path in mistral_files:
        if os.path.exists(file_path):
            print(f"처리 중: {file_path}")
            prompts = extract_prompts_from_jsonl(file_path)
            all_prompts.extend(prompts)
            print(f"  - {len(prompts)}개 프롬프트 추출됨")
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print(f"\n총 {len(all_prompts)}개 프롬프트 추출됨")
    
    # 도메인별로 분리하여 저장
    domains = ["General", "Legal", "Medical", "Technical"]
    total_saved = 0
    
    for domain in domains:
        count = save_domain_prompts(all_prompts, domain, output_dir)
        total_saved += count
    
    print(f"\n총 {total_saved}개 프롬프트가 {output_dir} 폴더에 저장되었습니다.")
    
    # 생성된 파일 목록 확인
    print(f"\n생성된 파일들:")
    for domain in domains:
        domain_dir = os.path.join(output_dir, domain.lower())
        if os.path.exists(domain_dir):
            for file in os.listdir(domain_dir):
                if file.endswith('.json'):
                    file_path = os.path.join(domain_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"  {domain_dir}/{file}: {len(data)}개 프롬프트")

if __name__ == "__main__":
    main() 