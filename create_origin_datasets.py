import json
import os
from collections import defaultdict

def clean_deepseek_response(response_text: str) -> str:
    """DeepSeek 모델의 <think> 태그를 제거합니다."""
    if not response_text:
        return response_text
    
    # <think> 태그 제거
    if response_text.startswith("<think>"):
        # <think> 태그 이후의 내용만 추출
        think_end = response_text.find("</think>")
        if think_end != -1:
            # </think> 태그 이후의 내용 추출
            response_text = response_text[think_end + 8:].strip()
        else:
            # </think> 태그가 없으면 <think> 태그만 제거
            response_text = response_text[7:].strip()
    
    return response_text

def extract_prompts_from_jsonl(file_path):
    """JSONL 파일에서 프롬프트만 추출"""
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # DeepSeek 모델의 <think> 태그 제거
                prompt_text = data["prompt"]
                if "deepseek" in data["model"].lower():
                    prompt_text = clean_deepseek_response(prompt_text)
                
                prompts.append({
                    "prompt": prompt_text,
                    "domain": data["domain"],
                    "model": data["model"],
                    "index": data["index"]
                })
            except json.JSONDecodeError:
                continue
    return prompts

def save_domain_prompts(prompts, domain, model_name, output_dir):
    """도메인별로 프롬프트를 JSON 파일로 저장"""
    domain_prompts = [p for p in prompts if p["domain"].lower() == domain.lower()]
    
    # 10000개로 제한
    domain_prompts = domain_prompts[:10000]
    
    # 모델별 폴더 생성
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 도메인별 하위 폴더 생성
    domain_dir = os.path.join(model_dir, domain.lower())
    os.makedirs(domain_dir, exist_ok=True)
    
    output_file = os.path.join(domain_dir, f"{domain.lower()}_prompts.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(domain_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"{model_name}/{domain}: {len(domain_prompts)}개 프롬프트 저장됨 -> {output_file}")
    return len(domain_prompts)

def main():
    # 모든 모델의 데이터 파일들
    data_files = [
        # mistral 데이터 파일들
        "dataset/mistral_7b/economy/mistral:7b_1000prompts_20250617_083304.jsonl",
        "dataset/mistral_7b/legal/mistral:7b_1000prompts_20250617_050705.jsonl",
        "dataset/mistral_7b/medical/mistral:7b_1000prompts_20250617_021449.jsonl",
        "dataset/mistral_7b/technical/mistral:7b_1000prompts_20250617_063930.jsonl",
        
        # deepseek 데이터 파일들 (존재하는 경우)
        "dataset/deepseek-r1_7b/economy/deepseek-r1:7b_1000prompts_20250617_083304.jsonl",
        "dataset/deepseek-r1_7b/legal/deepseek-r1:7b_1000prompts_20250617_050705.jsonl",
        "dataset/deepseek-r1_7b/medical/deepseek-r1:7b_1000prompts_20250617_021449.jsonl",
        "dataset/deepseek-r1_7b/technical/deepseek-r1:7b_1000prompts_20250617_063930.jsonl"
    ]
    
    output_dir = "dataset/origin"
    
    all_prompts = []
    
    # 모든 파일에서 프롬프트 추출
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"처리 중: {file_path}")
            prompts = extract_prompts_from_jsonl(file_path)
            all_prompts.extend(prompts)
            print(f"  - {len(prompts)}개 프롬프트 추출됨")
        else:
            print(f"파일을 찾을 수 없음: {file_path}")
    
    print(f"\n총 {len(all_prompts)}개 프롬프트 추출됨")
    
    # 모델별로 그룹화
    model_prompts = defaultdict(list)
    for prompt in all_prompts:
        model_prompts[prompt["model"]].append(prompt)
    
    # 도메인별로 분리하여 저장
    domains = ["Economy", "Legal", "Medical", "Technical"]
    total_saved = 0
    
    for model_name, prompts in model_prompts.items():
        print(f"\n=== {model_name} 모델 처리 중 ===")
        for domain in domains:
            count = save_domain_prompts(prompts, domain, model_name, output_dir)
            total_saved += count
    
    print(f"\n총 {total_saved}개 프롬프트가 {output_dir} 폴더에 저장되었습니다.")
    
    # 생성된 파일 목록 확인
    print(f"\n생성된 파일들:")
    for model_name in model_prompts.keys():
        model_dir = os.path.join(output_dir, model_name)
        if os.path.exists(model_dir):
            print(f"\n  {model_name}:")
            for domain in domains:
                domain_dir = os.path.join(model_dir, domain.lower())
                if os.path.exists(domain_dir):
                    for file in os.listdir(domain_dir):
                        if file.endswith('.json'):
                            file_path = os.path.join(domain_dir, file)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                print(f"    {domain_dir}/{file}: {len(data)}개 프롬프트")

if __name__ == "__main__":
    main() 