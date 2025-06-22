import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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
                    # 파일명에서 타임스탬프와 모델명 추출
                    timestamp = filename.split('_')[0] + '_' + filename.split('_')[1]
                    model_name = filename.split('_')[2].replace('.json', '')
                    
                    for result in data:
                        result['timestamp'] = timestamp
                        result['model_name'] = model_name
                        all_results.append(result)
            except Exception as e:
                print(f"파일 {filename} 로드 중 오류: {e}")
    
    return all_results

def analyze_domain_performance(results):
    """도메인별 성능을 분석합니다."""
    df = pd.DataFrame(results)
    
    # 도메인별 통계
    domain_stats = df.groupby('domain').agg({
        'avg_evidence_attention': ['mean', 'std', 'min', 'max'],
        'max_head': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("=== 도메인별 성능 분석 ===")
    print(domain_stats)
    print()
    
    # 도메인별 평균 evidence attention 시각화
    plt.figure(figsize=(10, 6))
    domain_means = df.groupby('domain')['avg_evidence_attention'].mean().sort_values(ascending=False)
    plt.bar(domain_means.index, domain_means.values)
    plt.title('Domain Performance by Average Evidence Attention')
    plt.ylabel('Average Evidence Attention')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('domain_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return domain_stats

def analyze_head_distribution(results):
    """헤드 분포를 분석합니다."""
    df = pd.DataFrame(results)
    
    print("=== 헤드 분포 분석 ===")
    
    # 전체 헤드 분포
    head_counts = df['max_head'].value_counts().sort_index()
    print("전체 헤드 분포:")
    print(head_counts)
    print()
    
    # 도메인별 헤드 분포
    print("도메인별 헤드 분포:")
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        head_dist = domain_df['max_head'].value_counts().sort_index()
        print(f"\n{domain}:")
        print(head_dist)
    
    # 헤드 분포 시각화
    domains = df['domain'].unique()
    n_domains = len(domains)
    
    # 서브플롯 개수 조정
    if n_domains <= 3:
        cols = 2
        rows = 2
    else:
        cols = 2
        rows = 3
    
    plt.figure(figsize=(15, 5*rows))
    
    # 전체 분포
    plt.subplot(rows, cols, 1)
    head_counts.plot(kind='bar')
    plt.title('Overall Head Distribution')
    plt.xlabel('Head Index')
    plt.ylabel('Frequency')
    
    # 도메인별 분포
    for i, domain in enumerate(domains):
        plt.subplot(rows, cols, i+2)
        domain_df = df[df['domain'] == domain]
        domain_df['max_head'].value_counts().sort_index().plot(kind='bar')
        plt.title(f'{domain} Domain Head Distribution')
        plt.xlabel('Head Index')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('head_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_attention_patterns(results):
    """어텐션 패턴을 분석합니다."""
    df = pd.DataFrame(results)
    
    print("=== 어텐션 패턴 분석 ===")
    
    # Evidence attention 값의 분포
    domains = df['domain'].unique()
    n_domains = len(domains)
    
    if n_domains <= 3:
        cols = 2
        rows = 2
    else:
        cols = 2
        rows = 3
    
    plt.figure(figsize=(15, 5*rows))
    
    # 전체 분포
    plt.subplot(rows, cols, 1)
    plt.hist(df['avg_evidence_attention'], bins=30, alpha=0.7)
    plt.title('Overall Evidence Attention Distribution')
    plt.xlabel('Evidence Attention')
    plt.ylabel('Frequency')
    
    # 도메인별 분포
    for i, domain in enumerate(domains):
        plt.subplot(rows, cols, i+2)
        domain_df = df[df['domain'] == domain]
        plt.hist(domain_df['avg_evidence_attention'], bins=20, alpha=0.7)
        plt.title(f'{domain} Domain Evidence Attention Distribution')
        plt.xlabel('Evidence Attention')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('attention_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 상관관계 분석
    correlation = df['avg_evidence_attention'].corr(df['max_head'])
    print(f"Evidence Attention과 Max Head 간의 상관계수: {correlation:.4f}")
    print()

def analyze_prompt_length_impact(results):
    """프롬프트 길이와 성능의 관계를 분석합니다."""
    df = pd.DataFrame(results)
    
    # 프롬프트 길이 계산
    df['prompt_length'] = df['prompt'].str.len()
    df['token_count'] = df['tokens'].apply(len)
    
    print("=== 프롬프트 길이 영향 분석 ===")
    
    # 길이별 평균 성능
    length_bins = pd.cut(df['prompt_length'], bins=5)
    length_performance = df.groupby(length_bins).agg({
        'avg_evidence_attention': 'mean',
        'max_head': 'mean'
    }).round(4)
    
    print("프롬프트 길이별 평균 성능:")
    print(length_performance)
    print()
    
    # 상관관계 분석
    length_corr = df['prompt_length'].corr(df['avg_evidence_attention'])
    token_corr = df['token_count'].corr(df['avg_evidence_attention'])
    
    print(f"프롬프트 길이와 Evidence Attention 상관계수: {length_corr:.4f}")
    print(f"토큰 수와 Evidence Attention 상관계수: {token_corr:.4f}")
    print()
    
    # 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['prompt_length'], df['avg_evidence_attention'], alpha=0.6)
    plt.xlabel('Prompt Length (characters)')
    plt.ylabel('Evidence Attention')
    plt.title('Prompt Length vs Evidence Attention')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['token_count'], df['avg_evidence_attention'], alpha=0.6)
    plt.xlabel('Token Count')
    plt.ylabel('Evidence Attention')
    plt.title('Token Count vs Evidence Attention')
    
    plt.tight_layout()
    plt.savefig('length_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_evidence_indices(results):
    """Evidence 인덱스 패턴을 분석합니다."""
    df = pd.DataFrame(results)
    
    print("=== Evidence 인덱스 패턴 분석 ===")
    
    # Evidence 인덱스의 위치 분석
    evidence_positions = []
    for indices in df['evidence_indices']:
        if indices:
            # 상대적 위치 (0~1 범위로 정규화)
            relative_positions = [i / (len(indices) - 1) if len(indices) > 1 else 0 for i in indices]
            evidence_positions.extend(relative_positions)
    
    plt.figure(figsize=(10, 6))
    plt.hist(evidence_positions, bins=20, alpha=0.7)
    plt.title('Evidence Token Relative Position Distribution')
    plt.xlabel('Relative Position (0=start, 1=end)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('evidence_positions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evidence 토큰 수 분석
    df['evidence_count'] = df['evidence_indices'].apply(len)
    
    print("Evidence 토큰 수 통계:")
    print(df['evidence_count'].describe())
    print()
    
    # Evidence 토큰 수와 성능의 관계
    evidence_count_corr = df['evidence_count'].corr(df['avg_evidence_attention'])
    print(f"Evidence 토큰 수와 Evidence Attention 상관계수: {evidence_count_corr:.4f}")
    print()

def generate_summary_report(results):
    """종합 요약 보고서를 생성합니다."""
    df = pd.DataFrame(results)
    
    print("=" * 60)
    print("실험 결과 종합 분석 보고서")
    print("=" * 60)
    print()
    
    # 기본 통계
    print(f"총 실험 수: {len(df)}")
    print(f"분석된 도메인: {', '.join(df['domain'].unique())}")
    print(f"사용된 모델: {', '.join(df['model_name'].unique())}")
    print()
    
    # 전체 성능 요약
    print("전체 성능 요약:")
    print(f"평균 Evidence Attention: {df['avg_evidence_attention'].mean():.4f}")
    print(f"Evidence Attention 표준편차: {df['avg_evidence_attention'].std():.4f}")
    print(f"최고 Evidence Attention: {df['avg_evidence_attention'].max():.4f}")
    print(f"최저 Evidence Attention: {df['avg_evidence_attention'].min():.4f}")
    print()
    
    # 도메인별 성능 순위
    domain_performance = df.groupby('domain')['avg_evidence_attention'].mean().sort_values(ascending=False)
    print("도메인별 성능 순위:")
    for i, (domain, score) in enumerate(domain_performance.items(), 1):
        print(f"{i}. {domain}: {score:.4f}")
    print()
    
    # 가장 활성화된 헤드
    head_usage = df['max_head'].value_counts()
    most_used_head = head_usage.index[0]
    print(f"가장 많이 사용된 헤드: {most_used_head} (사용 횟수: {head_usage.iloc[0]})")
    print()
    
    # 주요 발견사항
    print("주요 발견사항:")
    print("1. Evidence Attention 값의 분포와 패턴")
    print("2. 도메인별 성능 차이")
    print("3. 헤드 사용 패턴")
    print("4. 프롬프트 길이와 성능의 관계")
    print("5. Evidence 토큰 위치 패턴")
    print()

def main():
    """메인 분석 함수"""
    print("실험 결과 분석을 시작합니다...")
    
    # 결과 로드
    results = load_experiment_results()
    
    if not results:
        print("분석할 실험 결과가 없습니다.")
        return
    
    print(f"총 {len(results)}개의 실험 결과를 로드했습니다.")
    print()
    
    # 각종 분석 수행
    analyze_domain_performance(results)
    analyze_head_distribution(results)
    analyze_attention_patterns(results)
    analyze_prompt_length_impact(results)
    analyze_evidence_indices(results)
    generate_summary_report(results)
    
    print("분석이 완료되었습니다. 생성된 차트 파일들을 확인해주세요.")

if __name__ == "__main__":
    main() 