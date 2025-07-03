import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import matplotlib.pyplot as plt
from scipy import stats

def extract_features_from_results(results_json):
    data = []
    for r in results_json:
        # head별 evidence attention 평균값을 feature로 사용
        # 실험 결과에 avg_evidence_attention_per_head 또는 avg_evidence_attention_whole(리스트)가 있으면 사용
        if 'avg_evidence_attention_per_head' in r:
            attention_vector = r['avg_evidence_attention_per_head']
        elif isinstance(r.get('avg_evidence_attention_whole'), list):
            attention_vector = r['avg_evidence_attention_whole']
        elif isinstance(r.get('avg_evidence_attention'), list):
            attention_vector = r['avg_evidence_attention']  # 기존 호환성 유지
        else:
            # fallback: max_head one-hot
            attention_vector = [0]*32
            if 0 <= r.get('max_head', -1) < 32:
                attention_vector[r['max_head']] = 1
        
        attention_vector = np.array(attention_vector)
        
        # 기본 32개 head attention 값
        basic_features = list(attention_vector)
        
        # 통계적 특성 추가
        stats_features = [
            attention_vector.mean(),      # 평균
            attention_vector.std(),       # 표준편차
            attention_vector.var(),       # 분산
            attention_vector.max(),       # 최대값
            attention_vector.min(),       # 최소값
            attention_vector.max() - attention_vector.min(),  # 범위
            np.percentile(attention_vector, 25),  # 1사분위수
            np.percentile(attention_vector, 50),  # 중앙값
            np.percentile(attention_vector, 75),  # 3사분위수
            np.percentile(attention_vector, 75) - np.percentile(attention_vector, 25),  # IQR
        ]
        
        # 상대적 비율 (정규화)
        total_attention = attention_vector.sum()
        if total_attention > 0:
            normalized_features = list(attention_vector / total_attention)
        else:
            normalized_features = [0] * 32
        
        # 순위 정보
        rank_features = list(np.argsort(attention_vector)[::-1])  # 내림차순 순위
        
        # 차분값 (인접한 head 간의 attention 차이)
        diff_features = list(np.diff(attention_vector))
        if len(diff_features) < 31:  # 32개 head면 31개 차분값
            diff_features.extend([0] * (31 - len(diff_features)))
        
        # 집중도 지표
        concentration_features = [
            attention_vector.max() / (attention_vector.mean() + 1e-8),  # 최대값/평균 비율
            attention_vector.max() / (attention_vector.std() + 1e-8),   # 최대값/표준편차 비율
            (attention_vector > attention_vector.mean()).sum(),         # 평균 이상인 head 수
            (attention_vector > attention_vector.mean() + attention_vector.std()).sum(),  # 평균+표준편차 이상인 head 수
        ]
        
        # 모든 feature 결합
        all_features = (basic_features + stats_features + normalized_features + 
                       rank_features + diff_features + concentration_features)
        
        data.append(all_features + [r['domain']])
    
    # 컬럼명 생성
    basic_cols = [f'head_{i}' for i in range(32)]
    stats_cols = ['mean', 'std', 'var', 'max', 'min', 'range', 'q25', 'median', 'q75', 'iqr']
    norm_cols = [f'norm_head_{i}' for i in range(32)]
    rank_cols = [f'rank_head_{i}' for i in range(32)]
    diff_cols = [f'diff_head_{i}' for i in range(31)]
    concentration_cols = ['max_mean_ratio', 'max_std_ratio', 'above_mean_count', 'above_std_count']
    
    all_cols = basic_cols + stats_cols + norm_cols + rank_cols + diff_cols + concentration_cols + ['domain']
    
    df = pd.DataFrame(data, columns=all_cols)
    return df

def show():
    st.title("🛠️ 도메인 분류기 학습 (고급 Feature)")
    st.markdown("""
    실험 결과 파일을 선택해서 도메인 분류기를 학습하고 저장합니다.
    
    **새로운 Feature 엔지니어링:**
    - 기본 32개 head attention 값
    - 통계적 특성 (평균, 표준편차, 분산, 최대/최소, 사분위수 등)
    - 정규화된 attention 비율
    - 순위 정보
    - 인접 head 간 차분값
    - 집중도 지표
    
    총 **133개 feature**를 사용합니다!
    """)

    # 실험 결과 폴더 탐색
    experiment_root = "experiment_results"
    if not os.path.exists(experiment_root):
        st.error("experiment_results 폴더가 없습니다.")
        return

    model_dirs = [d for d in os.listdir(experiment_root) if os.path.isdir(os.path.join(experiment_root, d))]
    if not model_dirs:
        st.error("실험 결과가 있는 모델 폴더가 없습니다.")
        return
    selected_model = st.selectbox("모델 선택", model_dirs)
    result_files = [f for f in os.listdir(os.path.join(experiment_root, selected_model)) if f.endswith('.json') and not f.endswith('_errors.json')]
    if not result_files:
        st.error("선택한 모델에 실험 결과 파일이 없습니다.")
        return
    selected_file = st.selectbox("실험 결과 파일 선택", result_files)

    # 모델 선택 옵션
    st.markdown("### 🤖 모델 선택")
    model_options = st.multiselect(
        "학습할 모델들을 선택하세요",
        ["RandomForest", "XGBoost", "SVM", "Neural Network"],
        default=["RandomForest", "XGBoost"]
    )

    if st.button("고급 분류기 학습 및 저장"):
        with st.spinner("Feature 추출 중..."):
            path = os.path.join(experiment_root, selected_model, selected_file)
            with open(path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            df = extract_features_from_results(results)
            
            st.success(f"✅ {len(df)}개 샘플, {len(df.columns)-1}개 feature 추출 완료!")
            
            # Feature 정보 표시
            st.markdown("### 📊 Feature 정보")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**총 샘플 수**: {len(df)}")
                st.write(f"**총 Feature 수**: {len(df.columns)-1}")
                st.write(f"**도메인 수**: {len(df['domain'].unique())}")
            with col2:
                st.write("**도메인별 샘플 수**:")
                domain_counts = df['domain'].value_counts()
                for domain, count in domain_counts.items():
                    st.write(f"- {domain}: {count}개")
        
        with st.spinner("모델 학습 중..."):
            # Feature와 타겟 분리
            feature_cols = [col for col in df.columns if col != 'domain']
            X = df[feature_cols].values
            y = df['domain'].values
            
            # 레이블 인코딩
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # 학습/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
            
            # 모델 학습 및 비교
            models = {}
            results_comparison = []
            
            if "RandomForest" in model_options:
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                # 예측 결과를 원래 레이블로 변환
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_rf_original = label_encoder.inverse_transform(y_pred_rf)
                report_rf = classification_report(y_test_original, y_pred_rf_original, output_dict=True)
                models["RandomForest"] = (rf, label_encoder)
                results_comparison.append({
                    "Model": "RandomForest",
                    "Accuracy": report_rf['accuracy'],
                    "Macro F1": report_rf['macro avg']['f1-score'],
                    "Weighted F1": report_rf['weighted avg']['f1-score']
                })
                
                # RandomForest 모델 저장 (label_encoder 포함)
                with open('domain_classifier_rf.pkl', 'wb') as f:
                    pickle.dump((rf, label_encoder), f)
            
            if "XGBoost" in model_options:
                try:
                    import xgboost as xgb
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        eval_metric='mlogloss'
                    )
                    xgb_model.fit(X_train, y_train)
                    y_pred_xgb = xgb_model.predict(X_test)
                    # 예측 결과를 원래 레이블로 변환
                    y_test_original = label_encoder.inverse_transform(y_test)
                    y_pred_xgb_original = label_encoder.inverse_transform(y_pred_xgb)
                    report_xgb = classification_report(y_test_original, y_pred_xgb_original, output_dict=True)
                    models["XGBoost"] = (xgb_model, label_encoder)
                    results_comparison.append({
                        "Model": "XGBoost",
                        "Accuracy": report_xgb['accuracy'],
                        "Macro F1": report_xgb['macro avg']['f1-score'],
                        "Weighted F1": report_xgb['weighted avg']['f1-score']
                    })
                    
                    # XGBoost 모델 저장 (label_encoder 포함)
                    with open('domain_classifier_xgb.pkl', 'wb') as f:
                        pickle.dump((xgb_model, label_encoder), f)
                except ImportError:
                    st.warning("XGBoost가 설치되지 않았습니다. `pip install xgboost`로 설치해주세요.")
            
            if "SVM" in model_options:
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                
                # SVM을 위해 데이터 정규화
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                svm.fit(X_train_scaled, y_train)
                y_pred_svm = svm.predict(X_test_scaled)
                # 예측 결과를 원래 레이블로 변환
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_svm_original = label_encoder.inverse_transform(y_pred_svm)
                report_svm = classification_report(y_test_original, y_pred_svm_original, output_dict=True)
                models["SVM"] = (svm, scaler, label_encoder)
                results_comparison.append({
                    "Model": "SVM",
                    "Accuracy": report_svm['accuracy'],
                    "Macro F1": report_svm['macro avg']['f1-score'],
                    "Weighted F1": report_svm['weighted avg']['f1-score']
                })
                
                # SVM 모델 저장 (scaler, label_encoder 포함)
                with open('domain_classifier_svm.pkl', 'wb') as f:
                    pickle.dump((svm, scaler, label_encoder), f)
            
            if "Neural Network" in model_options:
                try:
                    from sklearn.neural_network import MLPClassifier
                    nn = MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.1
                    )
                    nn.fit(X_train, y_train)
                    y_pred_nn = nn.predict(X_test)
                    # 예측 결과를 원래 레이블로 변환
                    y_test_original = label_encoder.inverse_transform(y_test)
                    y_pred_nn_original = label_encoder.inverse_transform(y_pred_nn)
                    report_nn = classification_report(y_test_original, y_pred_nn_original, output_dict=True)
                    models["Neural Network"] = (nn, label_encoder)
                    results_comparison.append({
                        "Model": "Neural Network",
                        "Accuracy": report_nn['accuracy'],
                        "Macro F1": report_nn['macro avg']['f1-score'],
                        "Weighted F1": report_nn['weighted avg']['f1-score']
                    })
                    
                    # Neural Network 모델 저장 (label_encoder 포함)
                    with open('domain_classifier_nn.pkl', 'wb') as f:
                        pickle.dump((nn, label_encoder), f)
                except Exception as e:
                    st.warning(f"Neural Network 학습 중 오류: {str(e)}")
            
            st.success("✅ 모든 모델 학습 완료!")
        
        # 결과 비교 표시
        st.markdown("### 📈 모델 성능 비교")
        comparison_df = pd.DataFrame(results_comparison)
        st.dataframe(comparison_df)
        
        # 최고 성능 모델 찾기
        if results_comparison:
            best_model = max(results_comparison, key=lambda x: x['Macro F1'])
            st.success(f"🏆 최고 성능 모델: {best_model['Model']} (Macro F1: {best_model['Macro F1']:.4f})")
        
        # 상세 결과 표시
        for model_name, model_tuple in models.items():
            st.markdown(f"### 📊 {model_name} 상세 결과")
            
            # 모델과 label_encoder 분리
            if isinstance(model_tuple, tuple):
                if len(model_tuple) == 2:  # (model, label_encoder)
                    model, label_encoder = model_tuple
                    scaler = None
                elif len(model_tuple) == 3:  # (model, scaler, label_encoder)
                    model, scaler, label_encoder = model_tuple
                else:
                    continue
            else:
                model = model_tuple
                label_encoder = None
                scaler = None
            
            # 예측 수행
            if scaler:
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            y_pred = model.predict(X_test_processed)
            
            # 예측 결과를 원래 레이블로 변환
            if label_encoder:
                y_test_original = label_encoder.inverse_transform(y_test)
                y_pred_original = label_encoder.inverse_transform(y_pred)
                report = classification_report(y_test_original, y_pred_original, output_dict=True)
            else:
                report = classification_report(y_test, y_pred, output_dict=True)
            
            st.json(report)
            
            # Feature 중요도 분석 (RandomForest, XGBoost만)
            if hasattr(model, 'feature_importances_'):
                st.markdown(f"#### 🔍 {model_name} Feature 중요도 (상위 20개)")
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                st.dataframe(feature_importance)
                
                # 중요도 시각화 (RandomForest만)
                if model_name == "RandomForest":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = feature_importance.head(10)
                    ax.barh(range(len(top_features)), top_features['importance'])
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['feature'])
                    ax.set_xlabel('Feature Importance')
                    ax.set_title(f'{model_name} - Top 10 Most Important Features')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Medical 도메인 특성 분석
                st.markdown(f"#### 🏥 Medical 도메인 특성 분석")
                medical_analysis = analyze_domain_characteristics(model, X_train, y_train, feature_cols, 'medical')
                st.markdown(medical_analysis)

def analyze_domain_characteristics(model, X_train, y_train, feature_cols, target_domain='medical'):
    """특정 도메인의 특성을 분석합니다."""
    analysis = []
    
    # 타겟 도메인과 다른 도메인 분리
    target_mask = (y_train == target_domain)
    other_mask = (y_train != target_domain)
    
    if not target_mask.any() or not other_mask.any():
        return f"⚠️ {target_domain} 도메인 분석을 위한 충분한 데이터가 없습니다."
    
    X_target = X_train[target_mask]
    X_other = X_train[other_mask]
    
    # 각 feature에 대해 t-test 수행
    significant_features = []
    for i, feature_name in enumerate(feature_cols):
        target_values = X_target[:, i]
        other_values = X_other[:, i]
        
        # t-test
        t_stat, p_value = stats.ttest_ind(target_values, other_values)
        
        if p_value < 0.05:  # 유의수준 5%
            target_mean = target_values.mean()
            other_mean = other_values.mean()
            effect_size = abs(target_mean - other_mean) / (target_values.std() + other_values.std()) * 0.5
            
            significant_features.append({
                'feature': feature_name,
                'target_mean': target_mean,
                'other_mean': other_mean,
                'difference': target_mean - other_mean,
                'p_value': p_value,
                'effect_size': effect_size
            })
    
    # 효과 크기로 정렬
    significant_features.sort(key=lambda x: x['effect_size'], reverse=True)
    
    analysis.append(f"## {target_domain.capitalize()} 도메인 특성 분석 결과")
    analysis.append(f"")
    analysis.append(f"**총 Feature 수**: {len(feature_cols)}")
    analysis.append(f"**유의한 Feature 수**: {len(significant_features)} (p < 0.05)")
    analysis.append(f"**{target_domain} 샘플 수**: {target_mask.sum()}")
    analysis.append(f"**다른 도메인 샘플 수**: {other_mask.sum()}")
    analysis.append(f"")
    
    if significant_features:
        analysis.append("### 🎯 가장 중요한 차이점 (상위 10개)")
        analysis.append("")
        analysis.append("| Feature | Medical 평균 | 다른 도메인 평균 | 차이 | p-value | 효과크기 |")
        analysis.append("|---------|-------------|----------------|------|---------|----------|")
        
        for i, feat in enumerate(significant_features[:10]):
            analysis.append(f"| {feat['feature']} | {feat['target_mean']:.4f} | {feat['other_mean']:.4f} | {feat['difference']:.4f} | {feat['p_value']:.4f} | {feat['effect_size']:.4f} |")
        
        analysis.append("")
        analysis.append("### 💡 해석")
        analysis.append("")
        
        # 가장 큰 차이를 보이는 feature 분석
        top_feature = significant_features[0]
        if top_feature['difference'] > 0:
            analysis.append(f"- **{top_feature['feature']}**: Medical 도메인이 다른 도메인보다 {top_feature['difference']:.4f} 높음")
        else:
            analysis.append(f"- **{top_feature['feature']}**: Medical 도메인이 다른 도메인보다 {abs(top_feature['difference']):.4f} 낮음")
        
        # Feature 타입별 분석
        head_features = [f for f in significant_features if f['feature'].startswith('head_')]
        stats_features = [f for f in significant_features if f['feature'] in ['mean', 'std', 'var', 'max', 'min', 'range', 'q25', 'median', 'q75', 'iqr']]
        norm_features = [f for f in significant_features if f['feature'].startswith('norm_head_')]
        
        analysis.append(f"")
        analysis.append("### 📊 Feature 타입별 분석")
        analysis.append(f"- **기본 Head Features**: {len(head_features)}개 유의함")
        analysis.append(f"- **통계적 Features**: {len(stats_features)}개 유의함")
        analysis.append(f"- **정규화 Features**: {len(norm_features)}개 유의함")
        
        if head_features:
            analysis.append(f"")
            analysis.append("#### 🧠 중요한 Head들")
            for feat in head_features[:5]:
                head_num = feat['feature'].replace('head_', '')
                analysis.append(f"- **Head {head_num}**: Medical 도메인에서 {feat['difference']:.4f} {'높음' if feat['difference'] > 0 else '낮음'}")
    
    else:
        analysis.append("⚠️ 유의한 차이를 보이는 feature가 없습니다.")
    
    return "\n".join(analysis) 