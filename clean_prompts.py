#!/usr/bin/env python3
"""
프롬프트 데이터 정리 스크립트
잘못된 프롬프트들을 필터링하고 정리합니다.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

def is_valid_prompt(prompt: str) -> bool:
    """프롬프트가 유효한지 검증합니다."""
    if not prompt or not prompt.strip():
        return False
    
    prompt = prompt.strip()
    
    # 너무 짧은 프롬프트 제외
    if len(prompt) < 10:
        return False
    
    # AI 자기대화나 메타 텍스트 제외
    invalid_starts = [
        'sure!', 'sure,', 'okay,', 'alright,', 'here is', 'here\'s',
        'question:', 'prompt:', 'generate', 'create', 'write'
    ]
    
    prompt_lower = prompt.lower()
    for invalid_start in invalid_starts:
        if prompt_lower.startswith(invalid_start):
            return False
    
    # 불완전한 문장 제외 (끝에 구두점이 없는 경우)
    if not prompt.endswith(('.', '!', '?')):
        return False
    
    # 중복된 단어나 패턴 제외
    words = prompt.split()
    if len(set(words)) < len(words) * 0.7:  # 70% 이상 중복 단어가 있으면 제외
        return False
    
    return True

def clean_prompt_text(prompt: str) -> str:
    """프롬프트 텍스트를 정리합니다."""
    if not prompt:
        return prompt
    
    # 앞뒤 공백 제거
    prompt = prompt.strip()
    
    # 앞뒤 따옴표 제거
    while (prompt.startswith('"') and prompt.endswith('"')) or \
          (prompt.startswith("'") and prompt.endswith("'")):
        prompt = prompt[1:-1].strip()
    
    # 불필요한 공백 정리
    prompt = re.sub(r'\s+', ' ', prompt)
    
    return prompt

def process_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일을 처리하고 유효한 프롬프트만 반환합니다."""
    valid_prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    original_prompt = data.get('prompt', '')
                    
                    # 프롬프트 정리
                    cleaned_prompt = clean_prompt_text(original_prompt)
                    
                    # 유효성 검증
                    if is_valid_prompt(cleaned_prompt):
                        data['prompt'] = cleaned_prompt
                        valid_prompts.append(data)
                    else:
                        print(f"Invalid prompt in {file_path.name} line {line_num}: {original_prompt[:50]}...")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in {file_path.name} line {line_num}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return valid_prompts

def clean_dataset_directory(dataset_dir: str = "dataset/origin"):
    """데이터셋 디렉토리의 모든 JSONL 파일을 정리합니다."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    total_files = 0
    total_prompts_before = 0
    total_prompts_after = 0
    
    # 모든 JSONL 파일 찾기
    jsonl_files = list(dataset_path.rglob("*.jsonl"))
    
    for file_path in jsonl_files:
        print(f"\nProcessing: {file_path}")
        
        # 원본 파일 백업
        backup_path = file_path.with_suffix('.jsonl.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"Backup created: {backup_path}")
        
        # 파일 처리
        valid_prompts = process_jsonl_file(file_path)
        
        # 통계
        with open(file_path, 'r', encoding='utf-8') as f:
            original_count = sum(1 for _ in f)
        
        print(f"Original prompts: {original_count}")
        print(f"Valid prompts: {len(valid_prompts)}")
        print(f"Removed: {original_count - len(valid_prompts)} prompts")
        
        # 정리된 데이터로 파일 덮어쓰기
        if valid_prompts:
            with open(file_path, 'w', encoding='utf-8') as f:
                for prompt_data in valid_prompts:
                    f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
            print(f"File updated: {file_path}")
        else:
            print(f"Warning: No valid prompts found in {file_path}")
        
        total_files += 1
        total_prompts_before += original_count
        total_prompts_after += len(valid_prompts)
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_files}")
    print(f"Total prompts before: {total_prompts_before}")
    print(f"Total prompts after: {total_prompts_after}")
    print(f"Total removed: {total_prompts_before - total_prompts_after}")
    print(f"Removal rate: {((total_prompts_before - total_prompts_after) / total_prompts_before * 100):.1f}%")

if __name__ == "__main__":
    print("Starting prompt cleaning process...")
    clean_dataset_directory()
    print("Prompt cleaning completed!") 