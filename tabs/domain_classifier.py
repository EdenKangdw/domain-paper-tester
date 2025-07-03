import streamlit as st
import pickle
import numpy as np
import torch
import glob

def load_classifier(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_head_feature(prompt, model, tokenizer):
    # evidence 추출 및 attention 계산 (실험과 동일하게)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = tuple(attn.cpu().numpy() for attn in outputs.attentions)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    last_attn = attentions[-1][0]  # (head, from_token, to_token)
    # evidence_indices 추출 (여기선 예시로 마지막 토큰)
    evidence_indices = [len(tokens)-1]
    # 가장 강하게 반응한 head 찾기
    head_scores = [last_attn[h, :, evidence_indices].mean() for h in range(last_attn.shape[0])]
    max_head = int(np.argmax(head_scores))
    feature = np.zeros(32)
    feature[max_head] = 1
    return feature

def show():
    st.title("🧑‍🔬 도메인 분류기 실험")
    st.markdown("프롬프트를 입력하면 도메인 분류 결과를 보여줍니다.")
    prompt = st.text_area("프롬프트 입력", height=100)
    # 여러 분류기 파일 중 선택
    clf_files = glob.glob('domain_classifier*.pkl')
    if not clf_files:
        st.error("분류기 파일(domain_classifier*.pkl)이 없습니다. 먼저 학습 탭에서 분류기를 만들어주세요.")
        return
    selected_clf = st.selectbox("분류기 파일 선택", clf_files, index=0)
    if st.button("도메인 예측"):
        if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
            st.error("먼저 모델을 로드하세요.")
            return
        model = st.session_state['model']
        tokenizer = st.session_state['tokenizer']
        try:
            clf = load_classifier(selected_clf)
        except Exception as e:
            st.error(f"분류기 로드 실패: {e}")
            return
        feature = get_head_feature(prompt, model, tokenizer)
        pred = clf.predict([feature])[0]
        proba = clf.predict_proba([feature])[0]
        st.success(f"예측 도메인: **{pred}**")
        st.bar_chart({c: p for c, p in zip(clf.classes_, proba)})
        st.info(f"사용한 분류기 파일: {selected_clf}") 