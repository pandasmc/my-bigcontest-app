import streamlit as st
import pandas as pd
import google.generativeai as genai
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 페이지 기본 설정
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="내 가게를 살리는 AI 비밀상담사",
    page_icon="💡",
    layout="wide",
)

# ----------------------------------------------------------------------
# 1. 군집 해설집 (딕셔너리) - (사용자 최종 분석 결과 반영)
# ----------------------------------------------------------------------
# (중요!) 군집 번호(0, 1, 2...)와 사용자님의 분석 내용이
# 정확히 일치하는지 엑셀/분석표를 보고 다시 한번 확인/수정하세요.

# 고객 유형 (WHO) 군집 (K=6으로 가정)
WHO_CLUSTER_DESCRIPTIONS = {
    0.0: "30-50대 남성 중심 군집 (남성 30-60대가 많이옴, 여성 20-30대가 가장 적음)",
    1.0: "50-60대 고연령층 중심 군집 (남성 50-60대, 여성 50-60대 비중이 높음)",
    2.0: "비교적 모든 고객이 고르게 분포된 군집",
    3.0: "남성 50-60대 초고령층 중심 군집 (다른 모든 고객층 비중이 특히 낮음)",
    4.0: "20-30대 젊은 층 중심 군집 (남성 20대/여성 20-30대 많음, 40-60대 거의 없음)",
    5.0: "남성 20-30대, 여성 30대 비율이 비교적 높음"
}

# 가게 경쟁력 (SIZE) 군집 (K=3)
SIZE_CLUSTER_DESCRIPTIONS = {
    0.0: "경쟁력 '하' 그룹 (다 낮은 편인데 매출순위비율 0.1 평균)",
    1.0: "경쟁력 '최상(1등)' 그룹 (매출, 고객, 순위 모두 1등)",
    2.0: "경쟁력 '중위권' 그룹"
}

# 고객 관계 (LOYALTY) 군집 (K=4로 가정)
LOYALTY_CLUSTER_DESCRIPTIONS = {
    0.0: "고객 이탈형 (재방문, 신규 모두 낮음, 1점대)",
    1.0: "밸런스형 (재방문, 신규 모두 보통, 3점대)",
    2.0: "신규 유입형 (신규가 재방문보다 높음)",
    3.0: "충성 단골형 (재방문이 신규보다 높음)"
}

# ----------------------------------------------------------------------
# 2. 데이터 로드
# ----------------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    """모든 군집 분석이 완료된 최종 데이터 파일을 불러옵니다."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        st.info(f"'{filepath}' 파일이 app.py와 동일한 폴더에 있는지 확인하세요.")
        return None

# --- 데이터 로드 실행 ---
data = load_data("최종데이터.csv")

# ----------------------------------------------------------------------
# 3. Gemini 프롬프트 생성 함수
# ----------------------------------------------------------------------
def generate_prompt(store_name, industry, open_date, close_date, who_desc, size_desc, loyalty_desc):
    """3가지 군집 설명을 모두 포함하여 Gemini 프롬프트를 생성합니다."""
    close_info = "현재 운영 중" if pd.isna(close_date) else f"폐업일: {close_date}"
    prompt = f"""
당신은 대한민국 소상공인을 위한 최고의 AI 전략 컨설턴트입니다.
지금부터 내가 제공하는 [가맹점 기본 정보]와 [3차원 입체 분석 결과]를 기반으로,
{store_name} 사장님을 위한 맞춤형 전략 리포트를 작성해주세요.

[가맹점 기본 정보]
- 가맹점명: {store_name}
- 업종: {industry}
- 개설일: {open_date}
- {close_info}

[3차원 입체 분석 결과]
1. 고객 유형 (WHO): {who_desc}
2. 가게 경쟁력 (SIZE): {size_desc}
3. 고객 관계 (LOYALTY): {loyalty_desc}

[리포트 작성 가이드라인]
1. 위 3가지 분석 결과를 종합적으로 고려하여, 사장님이 지금 당장 실행해야 할 가장 중요하고 창의적인 '핵심 액션 플랜 1가지'를 제안해주세요.
2. 해당 액션 플랜을 실행했을 때의 '예상 기대효과'를 구체적인 수치를 포함하여 제시해주세요.
3. 전체적인 톤은 사장님께 직접 조언하는 것처럼 친절하고, 격려하며, 이해하기 쉬운 용어를 사용해주세요.
4. 리포트의 마지막에는 응원의 메시지를 추가해주세요.
"""
    return prompt.strip()

# ----------------------------------------------------------------------
# 4. Streamlit 화면 구성 (메인 로직)
# ----------------------------------------------------------------------

if data is None:
    st.stop()

st.title("💡 내 가게를 살리는 AI 비밀상담사")
st.caption("가맹점별 3차원 입체 진단 및 맞춤 전략 제안")

# --- 가맹점 검색 및 선택 ---
search = st.text_input("🔍 가맹점명을 입력하여 검색하세요 (예: 'A플러스 식당')", "")

filtered = data[data['가맹점명'].astype(str).str.contains(search, case=False, na=False)] if search else data.head(100)

if not filtered.empty:
    selected = st.selectbox("가맹점을 선택하세요", filtered['가맹점명'].unique())
    store_data = data[data['가맹점명'] == selected].iloc[0]

    # --- 3가지 군집 번호 및 텍스트 설명 가져오기 (핵심 결합 로직) ---
    try:
        # (중요!) final_master_data.csv에 저장된 "군집 컬럼명"과 일치해야 합니다.
        who_num = store_data.get('군집_고객유형_KM', None)
        size_num = store_data.get('군집_가게경쟁력_KM', None)
        loyalty_num = store_data.get('군집_고객관계_KM', None)

        who_desc = WHO_CLUSTER_DESCRIPTIONS.get(who_num, f"분석중 (군집번호: {who_num})")
        size_desc = SIZE_CLUSTER_DESCRIPTIONS.get(size_num, f"분석중 (군집번호: {size_num})")
        loyalty_desc = LOYALTY_CLUSTER_DESCRIPTIONS.get(loyalty_num, f"분석중 (군집번호: {loyalty_num})")
    except KeyError as e:
        st.error(f"오류: '{e}' 컬럼을 'final_master_data.csv'에서 찾을 수 없습니다.")
        st.stop()

    # --- 3가지 군집 진단 결과 보여주기 ---
    st.divider()
    st.header(f"'{selected}' 3차원 입체 진단")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("① 고객 유형 (WHO)")
        st.info(f"**{who_desc}**")
    with col2:
        st.subheader("② 가게 경쟁력 (SIZE)")
        st.success(f"**{size_desc}**")
    with col3:
        st.subheader("③ 고객 관계 (LOYALTY)")
        st.warning(f"**{loyalty_desc}**")

    # --- Gemini AI 리포트 생성 ---
    st.divider()
    st.header("🤖 AI 비밀상담사의 맞춤 전략 리포트")

    prompt = generate_prompt(
        store_name=store_data['가맹점명'],
        industry=store_data['업종_대분류'],
        open_date=store_data['개설일'],
        close_date=store_data['폐업일'],
        who_desc=who_desc,
        size_desc=size_desc,
        loyalty_desc=loyalty_desc
    )

    with st.expander("생성된 Gemini 프롬프트 보기"):
        st.text_area("프롬프트 내용", prompt, height=300)

    api_key = st.text_input("🔑 Google API 키를 입력하세요.", type="password")
    
    if st.button("🚀 AI 전략 리포트 생성하기"):
        if not api_key:
            st.error("Google API 키를 입력해주세요.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                with st.spinner("Gemini 2.5 Flash가 데이터를 분석하고 있습니다..."):
                    response = model.generate_content(prompt)
                
                st.subheader("💡 최종 분석 결과")
                with st.chat_message("ai"):
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했습니다: {e}")

elif search:
    st.warning("해당 가맹점을 찾을 수 없습니다. 검색어를 확인해주세요.")
