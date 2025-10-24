import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import warnings
import re
import json
import time
import io
import base64
import streamlit.components.v1 as components

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 한글 폰트 설정 (Streamlit Cloud 호환)
# ----------------------------------------------------------------------
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------------------------------------------------
# 1. 페이지 기본 설정
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="내 가게를 살리는 AI 비밀상담사",
    page_icon="💡",
    layout="wide",
)


# ----------------------------------------------------------------------
# 2. 헬퍼 함수 (데이터 로드, 프롬프트, 포맷팅 등)
# ----------------------------------------------------------------------
@st.cache_data
def load_data(filepath):
    """데이터를 로드하고, 표시용 리스트와 매핑용 딕셔너리를 반환합니다."""
    try:
        df = pd.read_csv(filepath, encoding='cp949')
        # 중복 제거 및 가나다 순 정렬
        unique_stores = sorted(df['가맹점명'].dropna().unique())
        
        # 표시용 리스트와, '표시 이름' -> '원본 이름' 매핑용 딕셔너리 생성
        display_list = [""] # Placeholder를 위한 빈 값
        display_to_original_map = {}
        
        for name in unique_stores:
            # 예: "본* (총 2글자)" 형식으로 표시 이름 생성
            display_name = f"{name} (총 {len(name)}글자)"
            display_list.append(display_name)
            display_to_original_map[display_name] = name
            
        return df, display_list, display_to_original_map
    except FileNotFoundError:
        st.error(f"오류: '{filepath}' 파일을 찾을 수 없습니다.")
        return None, None, None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None

# ----------------------------------------------------------------------
# 3. 맞춤형 설명 분석(Parsing) 및 프롬프트 생성 함수
# ----------------------------------------------------------------------
def parse_full_description(full_desc):
    """"맞춤형설명" 컬럼의 긴 텍스트를 파싱하여 딕셔너리로 반환합니다."""
    parsed_data = {
        "폐업 위험도": "데이터 없음", "주요 원인": "데이터 없음",
        "고객유형": "데이터 없음", "경쟁력": "데이터 없음", "고객관계": "데이터 없음"
    }
    if pd.isna(full_desc):
        return parsed_data
    try:
        pattern = re.compile(
            r"폐업 위험도:\s*(.*?)\.\s*주요 원인:\s*(.*?)\.\s*고객유형:\s*(.*?),\s*경쟁력:\s*(.*?),\s*고객관계:\s*(.*)",
            re.DOTALL
        )
        match = pattern.search(full_desc)
        if match:
            parsed_data["폐업 위험도"] = match.group(1).strip()
            parsed_data["주요 원인"] = match.group(2).strip()
            parsed_data["고객유형"] = match.group(3).strip()
            parsed_data["경쟁력"] = match.group(4).strip()
            parsed_data["고객관계"] = match.group(5).strip()
        return parsed_data
    except Exception as e:
        st.error(f"맞춤형 설명 분석 중 오류 발생: {e}")
        return parsed_data

def generate_prompt(store_name, industry, open_date, close_date, 
                    closure_risk, closure_factors, 
                    customer_type, competitiveness, customer_relation,
                    local_area_info,
                    trend_analysis_text):
    """AI에게 JSON 형식으로 구조화된 답변을 요청하는 프롬프트를 생성합니다."""
    close_info = "현재 운영 중" if pd.isna(close_date) else f"폐업일: {close_date}"
    prompt = f"""
당신은 대한민국 소상공인을 위한 최고의 AI 전략 컨설턴트입니다.
지금부터 내가 제공하는 정보를 종합적으로 분석하여 {store_name} 사장님을 위한
맞춤형 전략 리포트를 JSON 형식으로 작성해주세요.

[가맹점 기본 정보]
- 가맹점명: {store_name}, 업종: {industry}, 개설일: {open_date}, {close_info}

[AI 정밀 진단 요약]
- 폐업 위험도: {closure_risk}, 주요 원인: {closure_factors}
- 고객 유형: {customer_type}, 가게 경쟁력: {competitiveness}, 고객 관계: {customer_relation}
- 현재 상권 현황: {local_area_info}

[주요 지표 3개월 추세]
{trend_analysis_text}

[리포트 작성 가이드라인 (JSON 형식)]
1. 반드시 아래와 같은 JSON 형식으로만 답변해주세요. JSON 외에 다른 텍스트를 포함하지 마세요.
2. 'store_summary': 사장님 가게 유형을 한 문장으로 정의해주세요.
3. 'risk_signal', 'opportunity_signal': 가장 중요한 위험/기회 신호 1가지씩을 넣어주세요.
4. 'action_plan_detail': 구체적인 액션 플랜 1가지를 제안해주세요.
5. 'fact_based_example': 위 'action_plan'과 유사한 전략으로 성공한 (사실 기반의) 타 업종 사례를 1~2줄로 요약해주세요.
6. 'example_source': 위 성공 사례의 신뢰도를 위해, 관련 뉴스 기사 등의 출처 URL을 포함해주세요.
   - [중요] 만약 확실하고 유효한 URL을 모른다면, 절대 URL을 지어내지 말고 "출처 없음"으로 응답해주세요.
7. 'action_table': [단계, 실행 방안, 예상 비용]을 포함하는 마크다운 테이블 텍스트를 생성해주세요.
8. 'expected_effect': 예상 기대효과를 구체적인 수치로 제시해주세요.
9. 'encouragement': 사장님을 위한 따뜻한 응원의 메시지를 넣어주세요.
10. 'local_event_recommendation': [현재 상권 현황] 정보를 바탕으로, 해당 지역(예: {local_area_info})에서 진행 중이거나 예정인 관련 행사를 1개 추천해주세요.
    - [중요] 이 정보는 당신의 학습 데이터 기반이며 실시간 웹 검색이 아닙니다. 확실한 정보(행사명, 내용, 출처 URL)가 있을 때만 추천해주세요.
    - 유효한 URL이 없다면, "source" 값은 "정보 없음"으로 응답하고 절대 URL을 지어내지 마세요.

{{
  "store_summary": "...", "risk_signal": "...", "opportunity_signal": "...",
  "action_plan_title": "핵심 액션 플랜: [제목]", "action_plan_detail": "[상세 설명]",
  "fact_based_example": "[성공 사례 요약]",
  "example_source": "https://www.example-news.com/article/123",
  "action_table": "| 단계 | 실행 방안 | 예상 비용 |\\n|---|---|---|\\n| 1단계 | OOO 실행 | 10만원 |",
  "expected_effect": "신규 고객 15% 증가", "encouragement": "...",
  "local_event_recommendation": {{ "title": "지역 행사 정보 없음", "details": "현재 학습된 데이터 내에서 추천할 만한 관련 지역 행사를 찾지 못했습니다.", "source": "정보 없음" }}
}}
"""
    return prompt.strip()

def format_value(value, unit="", default_text="--"):
    """st.metric 값을 포맷팅합니다."""
    if pd.isna(value):
        return f"{default_text}{unit if unit == '%' else ''}"
    if unit == "%": return f"{value:.1f}%"
    if unit == "구간": return f"{int(value)} {unit}"
    return f"{value:.1f}"

# ----------------------------------------------------------------------
# 4. 추세 아이콘 생성 함수
# ----------------------------------------------------------------------
def format_trend_with_arrows(trend_value):
    """'증가 감소' 같은 텍스트를 두 줄의 시각적 HTML로 변환합니다."""
    if pd.isna(trend_value) or trend_value == "":
        return ""

    trend_map = {
        "증가": "<span style='color:red; font-weight:bold; font-size:1.1em;'>🔺 증가</span>",
        "감소": "<span style='color:blue; font-weight:bold; font-size:1.1em;'>🔻 감소</span>",
        "유지": "<span style='color:green; font-weight:bold; font-size:1.1em;'>➖ 유지</span>"
    }
    
    parts = trend_value.split(' ')

    if len(parts) == 1:
        trend1 = trend_map.get(parts[0], parts[0])
        return f"1개월 전 대비: {trend1}<br>2개월 전 대비: {trend1}"
    elif len(parts) == 2:
        trend1 = trend_map.get(parts[0], parts[0])
        trend2 = trend_map.get(parts[1], parts[1])
        return f"1개월 전 대비: {trend1}<br>2개월 전 대비: {trend2}"
    
    return trend_value

# ----------------------------------------------------------------------
# 5. 차트 생성 헬퍼 함수
# ----------------------------------------------------------------------
def plot_line_chart(ax, months, data_series, labels, title, colors, markers):
    """반복적인 선 그래프 생성 로직을 처리하는 함수"""
    for data, label, color, marker in zip(data_series, labels, colors, markers):
        ax.plot(months, data, label=label, color=color, marker=marker)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

def plot_bar_chart(ax, x, months, data_series, labels, title, colors):
    """반복적인 막대 그래프 생성 로직을 처리하는 함수"""
    bar_width = 0.4
    ax.bar(x, data_series[0], label=labels[0], width=bar_width, color=colors[0])
    ax.bar([i + bar_width for i in x], data_series[1], label=labels[1], width=bar_width, color=colors[1])
    ax.set_title(title, fontsize=12)
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(months, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# ----------------------------------------------------------------------
# 6. UI 구성 함수 (리포트, 홈페이지)
# ----------------------------------------------------------------------
def show_report(store_data, data):
    """상세 리포트 화면을 그립니다."""
    
    # [수정] UI/UX 개선을 위한 맞춤형 CSS
    st.markdown("""
    <style>
    /* ---------------------------------- */
    /* 1. 기본/메트릭 (테마 호환) */
    /* ---------------------------------- */
    .stApp {
        background-color: var(--background-color);
    }
    .metric-box {
        border-radius: 10px; padding: 15px;
        text-align: center; height: 100%;
        display: flex; flex-direction: column; justify-content: center;
        border: 1px solid var(--gray-30);
        background-color: var(--secondary-background-color); /* [수정] 테마 호환 */
        transition: box-shadow 0.3s ease-in-out;
    }
    .metric-box:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .metric-label { 
        font-size: 0.9em; 
        color: var(--gray-70); /* [수정] 테마 호환 */
        margin-bottom: 8px; font-weight: bold; 
    }
    .metric-value { 
        font-size: 1.5em; font-weight: 600; 
        color: var(--text-color); /* [수정] 테마 호환 */
        word-wrap: break-word; margin-bottom: 8px; 
    }
    .metric-trend { font-size: 0.9em; line-height: 1.5; }
    
    .box-color-1 { border-left: 5px solid #85C1E9; } 
    .box-color-2 { border-left: 5px solid #82E0AA; } 
    .box-color-3 { border-left: 5px solid #F7DC6F; } 
    .box-color-4 { border-left: 5px solid #F0B27A; } 
    .box-color-5 { border-left: 5px solid #D7BDE2; } 
    .box-color-6 { border-left: 5px solid #A3E4D7; } 

    /* ---------------------------------- */
    /* 2. 폐업 위험도 (라이트 모드) */
    /* ---------------------------------- */
    .risk-container {
        border-radius: 10px; padding: 20px;
        display: flex; align-items: center;
        border: 1px solid var(--gray-30);
    }
    .risk-level {
        flex: 2;
        font-weight: bold; font-size: 1.2em; text-align: center;
        padding: 1rem; border-radius: 0.5rem;
    }
    .risk-factors {
        flex: 5;
        padding-left: 20px;
        border-left: 1px solid var(--gray-30);
    }
    /* 라이트모드 기본값 */
    .risk-low { color: #0050b3; background-color: #e6f7ff; }
    .risk-high { color: #a8071a; background-color: #fff1f0; }
    .risk-medium { color: #237804; background-color: #f6ffed; }
    .risk-default { color: #595959; background-color: #fafafa; }

    /* ---------------------------------- */
    /* 3. 상권 현황 (테마 호환) */
    /* ---------------------------------- */
    .bar-chart-container { border: 1px solid var(--gray-30); border-radius: 10px; padding: 20px; }
    .bar-chart-header { display: flex; font-weight: bold; color: var(--gray-70); margin-bottom: 10px; }
    .bar-chart-row { display: flex; align-items: center; margin-bottom: 8px; font-size: 0.9em; }
    .bar-chart-label { flex: 2; text-align: left; padding-right: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bar-chart-bar-container { flex: 5; background-color: var(--gray-20); border-radius: 5px; }
    .bar-chart-bar { background-color: #5c9ce5; height: 20px; border-radius: 5px; }
    
    /* ---------------------------------- */
    /* 4. 차트 확대 효과 (기존과 동일) */
    /* ---------------------------------- */
    .zoom-chart {
      transition: transform 0.2s ease-in-out; 
      cursor: zoom-in;
    }
    .zoom-chart:hover {
      transform: scale(1.15); 
      z-index: 10; position: relative; 
      box-shadow: 0 8px 16px rgba(0,0,0,0.2);
      border-radius: 5px;
    }

    /* ---------------------------------- */
    /* 5. 탭 스타일 (라이트 모드) */
    /* ---------------------------------- */
    div[data-testid="stTabs"] button {
        background-color: transparent !important; 
        border: none !important;                 
        border-radius: 8px !important;           
        padding-top: 0.5em !important;    
        padding-bottom: 0.5em !important; 
        padding-left: 0.75em !important;  
        padding-right: 0.75em !important; 
        margin-right: 5px !important;     
        transition: transform 0.2s ease-in-out, background-color 0.2s;
    }
    div[data-testid="stTabs"] button > div {
        font-size: 1.2em !important;   
        font-weight: bold !important;
        color: var(--gray-70) !important; /* [수정] 테마 호환 */
    }
    div[data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
        transform: scale(1.1); 
        background-color: var(--gray-20) !important; /* [수정] 테마 호환 */
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #E6E6FA !important;    /* 라이트: 연보라 배경 */
        border: 1px solid #D8BFD8 !important; 
        border-radius: 8px !important;           
        transform: none; 
    }
    div[data-testid="stTabs"] button[aria-selected="true"] > div {
        color: #4B0082 !important; /* 라이트: 진보라 글씨 */
    }
    div[data-testid="stTabs"] > div:first-child {
       border-bottom: 2px solid var(--gray-30);
       margin-bottom: 10px;
    }

    /* ---------------------------------- */
    /* 6. 제목/탭 간격 (기존과 동일) */
    /* ---------------------------------- */
    div[data-testid="stTitle"] {
      margin-bottom: 25px !important; 
    }
    div[data-testid="stTabs"] {
      margin-bottom: 25px !important; 
    }

    /* ---------------------------------- */
    /* 7. [!!!핵심!!!] 다크 모드 오버라이드 */
    /* ---------------------------------- */
    @media (prefers-color-scheme: dark) {
        /* 다크모드일 때 .metric-box 테두리 */
        .metric-box {
            border: 1px solid var(--gray-70);
        }

        /* 다크모드일 때 폐업 위험도 색상 반전 */
        .risk-low { color: #91d5ff; background-color: #111a2c; }
        .risk-high { color: #ffa39e; background-color: #2c1618; }
        .risk-medium { color: #b7eb8f; background-color: #1a2b16; }
        .risk-default { color: #fafafa; background-color: #262730; }

        /* 다크모드일 때 탭 버튼 색상 반전 */
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #4B0082 !important;    /* 다크: 진보라 배경 */
            border: 1px solid #E6E6FA !important; 
        }
        div[data-testid="stTabs"] button[aria-selected="true"] > div {
            color: #E6E6FA !important; /* 다크: 연보라 글씨 */
        }
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("⬅️ 다른 가게 검색하기"):
        st.session_state.selected_store = None
        st.session_state.ai_report_data = None
        st.rerun()

    st.title(f"💡 '{st.session_state.selected_store}' 경영 진단 리포트")

    full_desc_string = store_data.get('맞춤형설명', None)
    parsed_data = parse_full_description(full_desc_string)

    tab1, tab2, tab3 = st.tabs(["🎯 AI 정밀 진단 (요약)", "📈 상세 데이터 (최근 3개월)", "🤖 AI 맞춤 전략 리포트"])

    with tab1:
        st.header("🎯 AI 정밀 진단 요약")
        st.markdown(f"**{store_data.get('업종', '업종정보 없음')}** 업종을 운영 중인 사장님 가게의 핵심 진단 결과입니다.")
        st.divider()

        st.subheader("🚨 폐업 위험도 분석")
        risk_level_text = parsed_data['폐업 위험도']
        css_class = "risk-default"
        if "낮음" in risk_level_text: css_class = "risk-low"
        elif "높음" in risk_level_text: css_class = "risk-high"
        elif "중간" in risk_level_text or "보통" in risk_level_text: css_class = "risk-medium"
        st.markdown(f"""
        <div class="risk-container">
            <div class="risk-level {css_class}">{risk_level_text}</div>
            <div class="risk-factors"><strong>주요 원인:</strong><br>{parsed_data['주요 원인']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.subheader("🧬 3차원 정밀 진단")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box box-color-1"><div class="metric-label">① 고객 유형</div><div class="metric-value">{parsed_data["고객유형"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box box-color-2"><div class="metric-label">② 가게 경쟁력</div><div class="metric-value">{parsed_data["경쟁력"]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-box box-color-3"><div class="metric-label">③ 고객 관계</div><div class="metric-value">{parsed_data["고객관계"]}</div></div>', unsafe_allow_html=True)
        st.divider()
        
        st.subheader("🏘️ 우리 상권 현황")
        current_district = store_data.get('상권') 
        if current_district and not pd.isna(current_district):
            district_df = data[data['상권'] == current_district]
            top_5_industries = district_df['업종'].value_counts().nlargest(5)
            if not top_5_industries.empty:
                st.write(f"**'{current_district}' 상권의 주요 업종 Top 5**")
                st.markdown('<div class="bar-chart-container">', unsafe_allow_html=True)
                st.markdown('<div class="bar-chart-header"><div class="bar-chart-label">업종</div><div style="flex: 5;">가게 수</div></div>', unsafe_allow_html=True)
                max_value = top_5_industries.max()
                for index, value in top_5_industries.items():
                    bar_width_percent = (value / max_value) * 100 if max_value > 0 else 0
                    st.markdown(f"""
                    <div class="bar-chart-row">
                        <div class="bar-chart-label">{index} ({value}개)</div>
                        <div class="bar-chart-bar-container">
                            <div class="bar-chart-bar" style="width: {bar_width_percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else: st.info(f"'{current_district}' 상권의 다른 업종 정보를 찾을 수 없습니다.")
        else: st.info("이 가게의 상권 정보 데이터를 찾을 수 없습니다.")
        st.divider()

        st.subheader("📊 주요 지표 최신 동향 (vs 3개월 전)")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            value = format_value(store_data.get('업종내매출순위비율_1m'), "%")
            trend = format_trend_with_arrows(store_data.get('업종내매출순위비율_추세'))
            st.markdown(f'<div class="metric-box box-color-4"><div class="metric-label">업종 내 매출 순위</div><div class="metric-value">{value}</div><div class="metric-trend">{trend}</div></div>', unsafe_allow_html=True)
        with metric_col2:
            value = format_value(store_data.get('재방문율_1m'), "%")
            trend = format_trend_with_arrows(store_data.get('재방문율_추세'))
            st.markdown(f'<div class="metric-box box-color-5"><div class="metric-label">재방문율</div><div class="metric-value">{value}</div><div class="metric-trend">{trend}</div></div>', unsafe_allow_html=True)
        with metric_col3:
            value = format_value(store_data.get('신규고객비율_1m'), "%")
            trend = format_trend_with_arrows(store_data.get('신규고객비율_추세'))
            st.markdown(f'<div class="metric-box box-color-6"><div class="metric-label">신규 고객 비율</div><div class="metric-value">{value}</div><div class="metric-trend">{trend}</div></div>', unsafe_allow_html=True)

    with tab2:
        st.header("📈 상세 시계열 추이 분석 (최근 3개월)")
        months = ['3개월 전', '2개월 전', '1개월 전']
        x = range(len(months))

        # --- 차트 크기/너비 설정 ---
        CHART_FIGSIZE = (6, 3.5) 
        CHART_WIDTH = 550        
        # ---------------------------

        st.subheader("고객 및 상권 동향")
        # --- [유지] 3칸, 작은 간격 ---
        chart_col1, chart_col2, chart_col3 = st.columns(3, gap="small") 
        
        with chart_col1:
            data_list = [store_data.get(f'유동고객비율_{m}m') for m in [3,2,1]] + [store_data.get(f'직장고객비율_{m}m') for m in [3,2,1]] + [store_data.get(f'거주고객비율_{m}m') for m in [3,2,1]]
            if pd.Series(data_list).notna().any():
                fig, ax = plt.subplots(figsize=CHART_FIGSIZE) 
                plot_line_chart(ax, months, [data_list[0:3], data_list[3:6], data_list[6:9]], ['유동고객', '직장고객', '거주고객'], "고객 유형 비율", ['steelblue', 'gray', 'darkgreen'], ['o', 's', '^'])
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode()
                # --- [유지] 왼쪽 정렬 ---
                st.markdown(f"<img src='data:image/png;base64,{img_data}' width='{CHART_WIDTH}' class='zoom-chart'>", unsafe_allow_html=True)
                plt.close(fig) 
            else: st.info("고객 유형 비율 데이터가 없습니다.")

        with chart_col2:
            data_list = [store_data.get(f'신규고객비율_{m}m') for m in [3,2,1]] + [store_data.get(f'재방문율_{m}m') for m in [3,2,1]]
            if pd.Series(data_list).notna().any():
                fig, ax = plt.subplots(figsize=CHART_FIGSIZE) 
                plot_line_chart(ax, months, [data_list[0:3], data_list[3:6]], ['신규고객', '재방문율'], "신규/재방문 고객", ['skyblue', 'salmon'], ['o', 's'])
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode()
                # --- [유지] 왼쪽 정렬 ---
                st.markdown(f"<img src='data:image/png;base64,{img_data}' width='{CHART_WIDTH}' class='zoom-chart'>", unsafe_allow_html=True)
                plt.close(fig) 
            else: st.info("신규/재방문 고객 데이터가 없습니다.")

        with chart_col3:
            data_list = [store_data.get(f'상권내폐업비율_{m}m') for m in [3,2,1]] + [store_data.get(f'업종내폐업비율_{m}m') for m in [3,2,1]]
            if pd.Series(data_list).notna().any():
                fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
                plot_line_chart(ax, months, [data_list[0:3], data_list[3:6]], ['상권내폐업', '업종내폐업'], "폐업 비율", ['gray', 'black'], ['o', 's'])
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode()
                # --- [유지] 왼쪽 정렬 ---
                st.markdown(f"<img src='data:image/png;base64,{img_data}' width='{CHART_WIDTH}' class='zoom-chart'>", unsafe_allow_html=True)
                plt.close(fig) 
            else: st.info("폐업 비율 데이터가 없습니다.")
            
        st.divider()
        st.subheader("매출 성과")
        # --- [수정] 윗줄과 동일하게 st.columns(3)으로 변경 ---
        chart_col4, chart_col5, _ = st.columns(3, gap="small") # 3번째 컬럼은 _ 로 받고 사용 안 함
        
        with chart_col4:
            data_list = [store_data.get(f'상권내매출순위비율_{m}m') for m in [3,2,1]] + [store_data.get(f'업종내매출순위비율_{m}m') for m in [3,2,1]]
            if pd.Series(data_list).notna().any():
                fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
                plot_bar_chart(ax, x, months, [data_list[0:3], data_list[3:6]], ['상권내', '업종내'], "매출 순위 비율 (상위 N%)", ['lightgray', 'steelblue'])
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode()
                # --- [유지] 왼쪽 정렬 ---
                st.markdown(f"<img src='data:image/png;base64,{img_data}' width='{CHART_WIDTH}' class='zoom-chart'>", unsafe_allow_html=True)
                plt.close(fig) 
            else: st.info("매출 순위 비율 데이터가 없습니다.")
            
        with chart_col5:
            data_list = [store_data.get(f'매출건수구간_{m}m') for m in [3,2,1]] + [store_data.get(f'매출금액구간_{m}m') for m in [3,2,1]]
            if pd.Series(data_list).notna().any():
                fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
                plot_bar_chart(ax, x, months, [data_list[0:3], data_list[3:6]], ['건수', '금액'], "매출 건수/금액 (구간)", ['gray', 'darkgreen'])
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode()
                # --- [유지] 왼쪽 정렬 ---
                st.markdown(f"<img src='data:image/png;base64,{img_data}' width='{CHART_WIDTH}' class='zoom-chart'>", unsafe_allow_html=True)
                plt.close(fig) 
            else: st.info("매출 건수/금액 데이터가 없습니다.")
    
    with tab3:
        st.header("🤖 AI 비밀상담사의 맞춤 전략 리포트")
        st.markdown("위의 AI 정밀 진단과 상세 데이터를 바탕으로 AI가 사장님만을 위한 맞춤 전략을 제안합니다.")
        def get_trend_str(col_name):
            val = store_data.get(col_name)
            return str(val) if not pd.isna(val) else "데이터 없음"
        trend_analysis_text = "\n".join([f"- {col.replace('_', ' ')}: {get_trend_str(col)}" for col in store_data.index if '추세' in col])
        
        local_info_for_prompt = "데이터 없음"
        if current_district and not pd.isna(current_district):
            district_df = data[data['상권'] == current_district]
            top_5_industries = district_df['업종'].value_counts().nlargest(5)
            if not top_5_industries.empty:
                local_info_for_prompt = ", ".join([f"{index} ({value}개)" for index, value in top_5_industries.items()])

        prompt = generate_prompt(
            store_name=store_data.get('가맹점명'), industry=store_data.get('업종'),
            open_date=store_data.get('개설일'), close_date=store_data.get('폐업일'),
            closure_risk=parsed_data['폐업 위험도'], closure_factors=parsed_data['주요 원인'],
            customer_type=parsed_data['고객유형'], competitiveness=parsed_data['경쟁력'],
            customer_relation=parsed_data['고객관계'],
            local_area_info=local_info_for_prompt, 
            trend_analysis_text=trend_analysis_text
        )

        if st.button("🚀 AI 전략 리포트 생성하기"):
            my_bar = st.progress(0, text="AI 분석을 시작합니다. 잠시만 기다려주세요...")
            try:
                for percent_complete in range(1, 81):
                    time.sleep(0.02)
                    text = "Gemini AI와 연결 중입니다..."
                    if percent_complete > 40: text = "사장님의 데이터를 안전하게 전송하고 있습니다..."
                    my_bar.progress(percent_complete, text=text)
                my_secret_key = st.secrets["GOOGLE_API_KEY"]
                genai.configure(api_key=my_secret_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                my_bar.progress(85, text="AI가 리포트를 생성하는 중입니다...")
                response = model.generate_content(prompt)
                my_bar.progress(95, text="AI의 답변을 분석하고 있습니다...")
                cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
                report_data = json.loads(cleaned_text)
                st.session_state.ai_report_data = report_data
                my_bar.progress(100, text="분석 완료!")
                time.sleep(1)
                my_bar.empty()
            except json.JSONDecodeError:
                my_bar.empty()
                st.error("AI가 JSON 형식으로 응답하지 않았습니다. 원본 응답을 표시합니다.")
                if 'response' in locals(): st.markdown(response.text)
                st.session_state.ai_report_data = None
            except Exception as e:
                my_bar.empty()
                st.error(f"AI 리포트 생성 중 오류 발생: {e}")
                st.session_state.ai_report_data = None

        if "ai_report_data" in st.session_state and st.session_state.ai_report_data:
            report_data = st.session_state.ai_report_data
            st.subheader("💡 최종 분석 결과")
            with st.chat_message("ai"):
                st.subheader("💬 사장님 가게 요약")
                st.info(report_data.get("store_summary", "요약 정보 없음"))
                st.subheader("🚦 위험 및 기회 신호")
                st.error(report_data.get("risk_signal", "위험 신호 없음"))
                st.success(report_data.get("opportunity_signal", "기회 신호 없음"))
                st.subheader(report_data.get("action_plan_title", "핵심 액션 플랜"))
                st.write(report_data.get("action_plan_detail", ""))
                
                st.subheader("💡 지역 연계 마케팅 제안")
                event_rec = report_data.get("local_event_recommendation", {})
                if event_rec and event_rec.get("title"):
                    st.success(f"**{event_rec.get('title')}**")
                    st.write(event_rec.get("details"))
                    source = event_rec.get("source")
                    if source and "http" in source:
                        st.caption(f"정보 출처: [{source}]({source})\n\n(참고: 위 출처는 AI가 생성한 예시 URL일 수 있으며, 실제 접속이 어려울 수 있습니다.)")
                else:
                    st.info("현재 추천할만한 주변 지역 행사를 찾지 못했습니다.")

                st.subheader("📚 참고: 유사 전략 성공 사례")
                st.warning(f"💡 {report_data.get('fact_based_example', '관련 사례 없음')}")
                source_url = report_data.get("example_source")
                if source_url and "http" in source_url:
                    st.caption(f"출처: [{source_url}]({source_url})\n\n(참고: 위 출처는 AI가 생성한 예시 URL일 수 있으며, 실제 접속이 어려울 수 있습니다.)")

                st.markdown(report_data.get("action_table", "실행 계획 없음"))
                st.subheader("📈 예상 기대효과")
                st.success(f'**목표:** {report_data.get("expected_effect", "데이터 없음")}')
                st.markdown("---")
                st.write(f"**AI 상담사의 응원 메시지:** {report_data.get('encouragement', '')}")

        with st.expander("AI에게 전달된 프롬MPT 내용 보기 (디버깅용)"):
            st.text_area("프롬프트 내용", prompt, height=300, disabled=True)

def show_homepage(display_list, display_to_original_map):
    """앱의 메인 화면(검색 페이지)을 그립니다."""
    st.markdown("<h1 style='text-align: center; color: var(--primary-color);'>💡 내 가게를 살리는 AI 비밀상담사</h1>", unsafe_allow_html=True)
    
    # 1. [수정] st.markdown 소제목을 삭제합니다. (HTML 안으로 이동)
    # st.markdown("<h3 style='text-align: center; color: var(--gray-70); margin-bottom: 0px;'>▼ 요즘 뜨는 키워드 ▼</h3>", unsafe_allow_html=True)

    # 2. 여기에 표시할 해시태그를 원하는 대로 수정하세요.
    hashtags = [
        "#성동구핫플",
        "#서울숲데이트",
        "#뚝섬맛집",
        "#성수동카페거리",
        "#요즘뜨는전시"
    ]
    
    # 3. 현재 Streamlit 테마('light' 또는 'dark')를 가져옵니다.
    current_theme = st.get_option("theme.base")

    # 4. components.html에 들어갈 내용
    html_content = f"""
    <html>
    <head>
        <style>
            html, body {{ 
                margin: 0; 
                padding: 0; 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif; /* Streamlit 폰트 적용 */
            }}
            
            /* [추가] 소제목 스타일 */
            .slider-subtitle {{
                text-align: center;
                font-weight: 600;
                font-size: 1.5em; /* h3와 유사한 크기 */
                color: #555; /* 라이트 모드 회색 */
                margin-bottom: 0px;
                margin-top: 5px;
            }}
            
            .hashtag-container {{
                text-align: center;
                height: 40px; 
                width: 100%;
                position: relative; 
                overflow: hidden;
                margin-top: 5px; /* 소제목과의 간격 */
                margin-bottom: 20px; /* 구분선과의 간격 */
            }}
            .hashtag-item {{
                font-size: 1.8em; 
                font-weight: bold; 
                color: #4B0082; /* 라이트 모드 기본 */
                position: absolute; 
                width: 100%; 
                left: 0;
                top: 0; 
                opacity: 0; 
                transition: opacity 0.5s ease-in-out; 
            }}
            .hashtag-item.active {{ 
                opacity: 1; 
            }}
            
            /* --- 다크 모드 CSS --- */
            body.dark .slider-subtitle {{
                color: #adbac7; /* 다크: 회색 */
            }}
            body.dark .hashtag-item {{
                color: #E6E6FA; /* 다크: 연보라 */
            }}
        </style>
    </head>
    <body class="{current_theme}">
        
        <h3 class="slider-subtitle">▼ 요즘 뜨는 키워드 ▼</h3>
        
        <div class="hashtag-container" id="hashtag-slider">
            </div>

        <script>
            // (스크립트 내용은 이전과 동일)
            function startHashtagSlider() {{
                if (window.hashtagSliderInitialized) return;
                const container = document.getElementById('hashtag-slider');
                if (!container) {{ setTimeout(startHashtagSlider, 300); return; }}
                
                window.hashtagSliderInitialized = true; 
                const tags = {json.dumps(hashtags)};
                let currentIndex = 0;

                tags.forEach((tag, index) => {{
                    const span = document.createElement('span');
                    span.className = 'hashtag-item';
                    span.textContent = tag;
                    if (index === 0) {{ span.classList.add('active'); }}
                    container.appendChild(span);
                }});

                const items = container.querySelectorAll('.hashtag-item');
                const totalItems = items.length;

                setInterval(() => {{
                    if(items[currentIndex]) {{ items[currentIndex].classList.remove('active'); }}
                    currentIndex = (currentIndex + 1) % totalItems;
                    if(items[currentIndex]) {{ items[currentIndex].classList.add('active'); }}
                }}, 2500); 
            }}
            startHashtagSlider();
        </script>
    </body>
    </html>
    """
    
    # 5. [수정] height를 100px로 넉넉하게 확보
    components.html(html_content, height=100)
    
    st.markdown("---") # 구분선

    selection = st.selectbox(
        "🔍 분석할 가게 이름을 검색하거나 목록에서 선택하세요.",
        options=display_list,
        placeholder="가게 이름의 앞부분을 입력하면 관련 목록이 나옵니다..."
    )

    st.info(
        "**검색 방법 안내**\n\n"
        "1. 검색창에 가게 이름의 **앞부분**을 입력하면 관련된 목록이 나타납니다. (예: `본죽`)\n"
        "2. 목록에서 사장님 가게의 **정확한 글자 수**를 확인하고 선택해주세요.\n\n"
        "--- \n"
        "**💡 왜 이름이 `***`로 나오나요?**\n\n"
        "데이터 개인정보 보호를 위해 가맹점명이 마스킹 처리되었습니다. "
        "글자 수로 구분하여 본인의 가게를 선택하시면 정확한 분석이 가능합니다."
    )

    if selection:
        if st.button(f"🚀 '{selection}' 경영 진단 리포트 보기"):
            original_masked_name = display_to_original_map[selection]
            st.session_state.selected_store = original_masked_name
            st.rerun()

# ----------------------------------------------------------------------
# 6. 메인 실행 로직
# ----------------------------------------------------------------------
def main():
    if 'selected_store' not in st.session_state:
        st.session_state.selected_store = None
        st.session_state.ai_report_data = None

    data, display_list, display_to_original_map = load_data("최종데이터.csv")
    if data is None:
        st.stop()

    if st.session_state.selected_store is None:
        show_homepage(display_list, display_to_original_map)
    else:
        try:
            store_data_row = data[data['가맹점명'] == st.session_state.selected_store].iloc[0]
            show_report(store_data_row, data)
        except (IndexError, KeyError) as e:
            st.error("선택한 가게 정보를 찾는 데 실패했습니다. 다시 검색해주세요.")
            st.session_state.selected_store = None
            if st.button("홈으로 돌아가기"):
                st.rerun()

if __name__ == "__main__":
    main()

