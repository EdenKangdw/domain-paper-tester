import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def load_experiment_results():
    """실험 결과 파일들을 로드합니다."""
    results_dir = "experiment_results"
    all_results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for result in data:
                        all_results.append(result)
            except Exception as e:
                print(f"파일 {filename} 로드 중 오류: {e}")
    
    return all_results

def analyze_domain_significant_heads(results):
    """도메인별 유의미한 어텐션 헤드를 분석합니다."""
    df = pd.DataFrame(results)
    
    print("=" * 80)
    print("도메인별 유의미한 어텐션 헤드 분석")
    print("=" * 80)
    print()
    
    # 전체 헤드 사용 통계
    print("📊 전체 헤드 사용 통계")
    total_head_usage = df['max_head'].value_counts().sort_values(ascending=False)
    print(f"총 사용된 헤드 수: {len(total_head_usage)}")
    print(f"가장 많이 사용된 헤드: {total_head_usage.index[0]} (사용 {total_head_usage.iloc[0]}회)")
    print()
    
    # 도메인별 분석
    domains = df['domain'].unique()
    
    for domain in domains:
        print(f"🔍 {domain.upper()} 도메인 분석")
        print("-" * 50)
        
        domain_df = df[df['domain'] == domain]
        domain_head_usage = domain_df['max_head'].value_counts().sort_values(ascending=False)
        
        # 기본 통계
        print(f"총 샘플 수: {len(domain_df)}")
        print(f"사용된 헤드 수: {len(domain_df['max_head'].unique())}")
        print(f"평균 Evidence Attention: {domain_df['avg_evidence_attention'].mean():.4f}")
        print()
        
        # 상위 5개 헤드
        print("🏆 상위 5개 유의미한 헤드:")
        for i, (head, count) in enumerate(domain_head_usage.head(5).items(), 1):
            percentage = (count / len(domain_df)) * 100
            avg_attention = domain_df[domain_df['max_head'] == head]['avg_evidence_attention'].mean()
            print(f"  {i}. 헤드 {head}: {count}회 사용 ({percentage:.1f}%), 평균 어텐션: {avg_attention:.4f}")
        
        # 주요 헤드 식별 (전체 대비 비율이 높은 헤드)
        print("\n🎯 도메인 특화 헤드 (전체 대비 비율이 높은 헤드):")
        domain_specific_heads = []
        for head in domain_head_usage.index:
            domain_ratio = domain_head_usage[head] / len(domain_df)
            total_ratio = total_head_usage.get(head, 0) / len(df)
            if total_ratio > 0:  # 전체에서 사용된 헤드만
                relative_ratio = domain_ratio / total_ratio
                if relative_ratio > 1.5:  # 1.5배 이상 높으면 특화 헤드로 간주
                    domain_specific_heads.append((head, relative_ratio, domain_head_usage[head]))
        
        domain_specific_heads.sort(key=lambda x: x[1], reverse=True)
        for head, ratio, count in domain_specific_heads[:3]:
            print(f"  헤드 {head}: 전체 대비 {ratio:.2f}배 높은 사용률, {count}회 사용")
        
        print("\n" + "=" * 80)
        print()

def analyze_head_domain_specialization(results):
    """헤드별 도메인 특화도를 분석합니다."""
    df = pd.DataFrame(results)
    
    print("🎯 헤드별 도메인 특화도 분석")
    print("=" * 80)
    print()
    
    # 각 헤드별로 도메인 분포 분석
    head_domain_dist = df.groupby(['max_head', 'domain']).size().unstack(fill_value=0)
    
    # 특화도 계산 (특정 도메인에서 집중적으로 사용되는 헤드)
    specialization_scores = {}
    
    for head in head_domain_dist.index:
        domain_counts = head_domain_dist.loc[head]
        total_usage = domain_counts.sum()
        
        if total_usage >= 5:  # 최소 5회 이상 사용된 헤드만 분석
            # 가장 많이 사용된 도메인
            max_domain = domain_counts.idxmax()
            max_count = domain_counts.max()
            
            # 특화도 = 해당 도메인에서의 사용 비율
            specialization = max_count / total_usage
            
            # 전체에서의 해당 도메인 비율
            total_domain_ratio = df[df['domain'] == max_domain].shape[0] / len(df)
            
            # 상대적 특화도 (전체 대비)
            relative_specialization = specialization / total_domain_ratio
            
            specialization_scores[head] = {
                'specialized_domain': max_domain,
                'usage_count': max_count,
                'total_usage': total_usage,
                'specialization_ratio': specialization,
                'relative_specialization': relative_specialization
            }
    
    # 특화도 순으로 정렬
    sorted_specialization = sorted(specialization_scores.items(), 
                                  key=lambda x: x[1]['relative_specialization'], 
                                  reverse=True)
    
    print("🏆 도메인 특화 헤드 순위 (상위 10개):")
    print()
    for i, (head, info) in enumerate(sorted_specialization[:10], 1):
        print(f"{i:2d}. 헤드 {head:2d}: {info['specialized_domain']:10s} 도메인 특화")
        print(f"     사용 횟수: {info['usage_count']}/{info['total_usage']} ({info['specialization_ratio']:.1%})")
        print(f"     상대적 특화도: {info['relative_specialization']:.2f}배")
        print()
    
    return sorted_specialization

def analyze_attention_strength_by_domain(results):
    """도메인별 어텐션 강도를 분석합니다."""
    df = pd.DataFrame(results)
    
    print("💪 도메인별 어텐션 강도 분석")
    print("=" * 80)
    print()
    
    # 도메인별 평균 어텐션 강도
    domain_attention = df.groupby('domain')['avg_evidence_attention'].agg(['mean', 'std', 'min', 'max']).round(4)
    
    print("📊 도메인별 Evidence Attention 통계:")
    print(domain_attention)
    print()
    
    # 각 도메인에서 가장 강한 어텐션을 보인 헤드들
    print("🔥 도메인별 최고 어텐션 헤드:")
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        max_attention_idx = domain_df['avg_evidence_attention'].idxmax()
        max_attention_row = domain_df.loc[max_attention_idx]
        
        print(f"\n{domain.upper()} 도메인:")
        print(f"  최고 어텐션: 헤드 {max_attention_row['max_head']} ({max_attention_row['avg_evidence_attention']:.4f})")
        print(f"  프롬프트: {max_attention_row['prompt'][:100]}...")
    
    print()

def generate_domain_head_summary(results):
    """도메인별 헤드 사용 요약을 생성합니다."""
    df = pd.DataFrame(results)
    
    print("📋 도메인별 헤드 사용 요약")
    print("=" * 80)
    print()
    
    summary = {}
    
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        head_usage = domain_df['max_head'].value_counts()
        
        # 주요 헤드 (사용률 10% 이상)
        total_samples = len(domain_df)
        major_heads = head_usage[head_usage >= total_samples * 0.1]
        
        # 평균 어텐션
        avg_attention = domain_df['avg_evidence_attention'].mean()
        
        summary[domain] = {
            'total_samples': total_samples,
            'unique_heads': len(head_usage),
            'major_heads': major_heads.to_dict(),
            'avg_attention': avg_attention,
            'most_used_head': head_usage.index[0] if len(head_usage) > 0 else None
        }
    
    # 요약 출력
    for domain, info in summary.items():
        print(f"🔹 {domain.upper()} 도메인:")
        print(f"   샘플 수: {info['total_samples']}")
        print(f"   사용된 헤드 수: {info['unique_heads']}")
        print(f"   평균 Evidence Attention: {info['avg_attention']:.4f}")
        print(f"   가장 많이 사용된 헤드: {info['most_used_head']}")
        
        if info['major_heads']:
            print(f"   주요 헤드 (10% 이상 사용):")
            for head, count in info['major_heads'].items():
                percentage = (count / info['total_samples']) * 100
                print(f"     헤드 {head}: {count}회 ({percentage:.1f}%)")
        print()

def main():
    """메인 분석 함수"""
    print("도메인별 유의미한 어텐션 헤드 분석을 시작합니다...")
    print()
    
    # 결과 로드
    results = load_experiment_results()
    
    if not results:
        print("분석할 실험 결과가 없습니다.")
        return
    
    print(f"총 {len(results)}개의 실험 결과를 로드했습니다.")
    print()
    
    # 각종 분석 수행
    analyze_domain_significant_heads(results)
    analyze_head_domain_specialization(results)
    analyze_attention_strength_by_domain(results)
    generate_domain_head_summary(results)
    
    print("분석이 완료되었습니다.")

if __name__ == "__main__":
    main() 