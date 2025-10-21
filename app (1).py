
import streamlit as st
import pandas as pd

df = pd.read_csv("최종_가맹점_진단결과.csv")

def generate_store_report(store_id: str, df: pd.DataFrame) -> str:
    store_df = df[df['가맹점ID'] == store_id].sort_values('기준년월')
    latest = store_df.iloc[-1]
    risk_score = latest['폐업위험도']
    alert_text = latest.get('위기경보_텍스트', '경보 없음')
    cluster_type = latest.get('군집_운영유형_KM', '정보 없음')

    report = f"""
    📍 가맹점명: {latest['가맹점명']}
    🧠 AI 폐업 위험도: {risk_score:.1f}%
    🚨 위기 경보: {alert_text}
    🧩 운영 유형 군집: {cluster_type}

    최근 3개월 평균 매출: {latest.get('Sales_3M_MA', '없음')}
    매출 변동성(표준편차): {latest.get('Sales_3M_Std', '없음')}
    고객수 평균: {latest.get('Customer_3M_MA', '없음')}

    AI 분석에 따르면, 이 가맹점은 '{cluster_type}' 유형에 속하며,
    폐업 위험도는 {risk_score:.1f}%로 평가되었습니다.
    {alert_text}
    """
    return report.strip()

st.title("🧠 AI 비서: 가맹점 진단 리포트")

search = st.text_input("🔍 가맹점명을 입력하세요")
filtered = df[df['가맹점명'].str.contains(search, case=False, na=False)]

if not filtered.empty:
    selected = st.selectbox("가맹점을 선택하세요", filtered['가맹점ID'].unique())
    if selected:
        report = generate_store_report(selected, df)
        st.markdown("## 🧾 AI 비서 리포트")
        st.text_area("리포트 내용", report, height=300)
else:
    st.warning("해당 가맹점을 찾을 수 없습니다.")
