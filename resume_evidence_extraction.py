#!/usr/bin/env python3
"""
Evidence 추출 진행 상황 저장 및 재개 스크립트
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class EvidenceExtractionProgress:
    def __init__(self, model_key: str, domain: str, total_prompts: int):
        self.model_key = model_key
        self.domain = domain
        self.total_prompts = total_prompts
        self.processed_indices = set()
        self.completed_data = []
        self.start_time = datetime.now()
        self.last_save_time = datetime.now()
        
    def add_completed(self, index: int, data: Dict[str, Any]):
        """완료된 프롬프트 추가"""
        self.processed_indices.add(index)
        self.completed_data.append(data)
        
    def get_progress(self) -> float:
        """진행률 반환 (0.0 ~ 1.0)"""
        return len(self.processed_indices) / self.total_prompts
        
    def get_remaining(self) -> List[int]:
        """남은 인덱스 목록 반환"""
        all_indices = set(range(self.total_prompts))
        return sorted(list(all_indices - self.processed_indices))
        
    def save_progress(self, filepath: str):
        """진행 상황 저장"""
        progress_data = {
            'model_key': self.model_key,
            'domain': self.domain,
            'total_prompts': self.total_prompts,
            'processed_indices': list(self.processed_indices),
            'completed_data': self.completed_data,
            'start_time': self.start_time.isoformat(),
            'last_save_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(progress_data, f)
        
        self.last_save_time = datetime.now()
        print(f"💾 진행 상황 저장됨: {filepath}")
        print(f"   진행률: {self.get_progress():.1%} ({len(self.processed_indices)}/{self.total_prompts})")
        
    @classmethod
    def load_progress(cls, filepath: str) -> 'EvidenceExtractionProgress':
        """저장된 진행 상황 로드"""
        with open(filepath, 'rb') as f:
            progress_data = pickle.load(f)
        
        progress = cls(
            progress_data['model_key'],
            progress_data['domain'],
            progress_data['total_prompts']
        )
        progress.processed_indices = set(progress_data['processed_indices'])
        progress.completed_data = progress_data['completed_data']
        progress.start_time = datetime.fromisoformat(progress_data['start_time'])
        progress.last_save_time = datetime.fromisoformat(progress_data['last_save_time'])
        
        print(f"📂 진행 상황 로드됨: {filepath}")
        print(f"   진행률: {progress.get_progress():.1%} ({len(progress.processed_indices)}/{progress.total_prompts})")
        
        return progress

def save_progress_file(progress: EvidenceExtractionProgress, model_key: str, domain: str):
    """진행 상황 파일 저장"""
    progress_dir = Path("progress")
    progress_dir.mkdir(exist_ok=True)
    
    safe_model_key = model_key.replace(":", "_")
    filename = f"{safe_model_key}_{domain}_progress.pkl"
    filepath = progress_dir / filename
    
    progress.save_progress(str(filepath))
    return filepath

def load_progress_file(model_key: str, domain: str) -> EvidenceExtractionProgress:
    """진행 상황 파일 로드"""
    progress_dir = Path("progress")
    safe_model_key = model_key.replace(":", "_")
    filename = f"{safe_model_key}_{domain}_progress.pkl"
    filepath = progress_dir / filename
    
    if filepath.exists():
        return EvidenceExtractionProgress.load_progress(str(filepath))
    else:
        return None

def check_existing_progress(model_key: str, domain: str) -> bool:
    """기존 진행 상황 확인"""
    progress = load_progress_file(model_key, domain)
    if progress:
        print(f"🔍 기존 진행 상황 발견:")
        print(f"   모델: {progress.model_key}")
        print(f"   도메인: {progress.domain}")
        print(f"   진행률: {progress.get_progress():.1%}")
        print(f"   완료된 데이터: {len(progress.completed_data)}개")
        return True
    return False

if __name__ == "__main__":
    # 사용 예시
    print("🔍 Evidence 추출 진행 상황 관리 도구")
    print("=" * 50)
    
    # 기존 진행 상황 확인
    model_key = "gemma:7b"
    domain = "economy"
    
    if check_existing_progress(model_key, domain):
        print("✅ 기존 진행 상황이 있습니다.")
    else:
        print("❌ 기존 진행 상황이 없습니다.") 