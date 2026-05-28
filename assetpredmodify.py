import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==========================================
# 1. 頁面設定與 iOS 風格 UI 初始化
# ==========================================
st.set_page_config(page_title="人生財務曲線預測", page_icon="📈", layout="wide")

st.title("📈 預測你的人生財務曲線")
st.markdown("透過調整左側的參數，視覺化評估**純儲蓄**、**資產投資**與**購屋決策**對未來總資產的影響。")

# ==========================================
# 2. 側邊欄參數設定 (Sidebar Inputs)
# ==========================================
with st.sidebar:
    st.header("⚙️ 參數設定")
    
    st.subheader("👤 基本財務狀況")
    起始年齡 = st.number_input("目前年齡", min_value=20, max_value=60, value=32, step=1)
    退休年齡 = st.number_input("預計退休年齡", min_value=40, max_value=80, value=65, step=1)
    起始資金 = st.number_input("起始資金 (萬元)", min_value=0, value=100, step=10)
    
    每月薪水 = st.number_input("平均每月薪水 (萬元)", min_value=0.0, value=8.0, step=0.5)
    每月開銷 = st.number_input("每月生活開銷 (萬元)", min_value=0.0, value=3.0, step=0.1, help="不含房租與房貸")
    每月房租 = st.number_input("每月房租 (萬元)", min_value=0.0, value=2.0, step=0.1)
    
    st.markdown("---")
    st.subheader("📈 投資策略")
    投資部位 = st.slider("可支配資金投資比例 (%)", min_value=0, max_value=100, value=70, step=5) / 100.0
    投資年利率 = st.number_input("預期年化報酬率 (%)", min_value=0.0, value=6.0, step=0.5) / 100.0
    
    st.markdown("---")
    st.subheader("🏠 購屋計畫")
    買房年齡 = st.number_input("預計買房年齡", min_value=起始年齡, max_value=80, value=40, step=1)
    買房價格 = st.number_input("房屋總價 (萬元)", min_value=100, value=1500, step=100)
    買房頭期款 = st.number_input("頭期款 (萬元)", min_value=100, value=300, step=50)
    貸款年數 = st.number_input("房貸年限 (年)", min_value=10, max_value=40, value=30, step=1)
    房貸利率 = st.number_input("房貸利率 (%)", min_value=1.0, value=2.1, step=0.1) / 100.0

# ==========================================
# 3. 核心財務計算邏輯
# ==========================================
預測時段 = np.arange(起始年齡, 100)
df = pd.DataFrame(index=預測時段)
df.index.name = '年齡'

# 計算年度基礎現金流 (無買房情況)
年薪 = np.where(df.index <= 退休年齡, 每月薪水 * 12, 0)
年開銷 = 每月開銷 * 12
年房租 = 每月房租 * 12
基礎年淨額 = 年薪 - 年開銷 - 年房租

# 計算買房相關現金流
# 房貸本息攤還試算 (簡化版年繳)
月利率 = 房貸利率 / 12
貸款總額 = 買房價格 - 買房頭期款
if 貸款總額 > 0 and 貸款年數 > 0:
    月繳房貸 = 貸款總額 * (月利率 * (1 + 月利率)**(貸款年數*12)) / ((1 + 月利率)**(貸款年數*12) - 1)
    年繳房貸 = 月繳房貸 * 12
else:
    月繳房貸 = 0
    年繳房貸 = 0

買房年淨額 = np.copy(基礎年淨額)
# 買房當年扣除頭期款
if 買房年齡 in df.index:
    idx_buy = np.where(df.index == 買房年齡)[0][0]
    買房年淨額[idx_buy] -= 買房頭期款
    
    # 買房後停止付房租，開始付房貸
    idx_mortgage_end = min(idx_buy + 貸款年數, len(df))
    # 買房後的年份加回原本扣掉的房租 (因為不用租了)
    買房年淨額[idx_buy:] += 年房租 
    # 扣除房貸
    買房年淨額[idx_buy:idx_mortgage_end] -= 年繳房貸

# 複利計算函式 $FV = PV \times (1 + r)$
def calculate_wealth(net_cash_flow, start_cap, inv_ratio, inv_rate):
    wealth = []
    current = start_cap
    for cf in net_cash_flow:
        if current > 0:
            invested = current * inv_ratio
            cash = current * (1 - inv_ratio)
            current = invested * (1 + inv_rate) + cash + cf
        else:
            # 資產為負時無法投資，只能靠現金流填補
            current = current + cf
        wealth.append(current)
    return wealth

# 產出四種情境的數據
df['無投資_無買房'] = calculate_wealth(基礎年淨額, 起始資金, 0, 0)
df['有投資_無買房'] = calculate_wealth(基礎年淨額, 起始資金, 投資部位, 投資年利率)
df['無投資_有買房'] = calculate_wealth(買房年淨額, 起始資金, 0, 0)
df['有投資_有買房'] = calculate_wealth(買房年淨額, 起始資金, 投資部位, 投資年利率)

# ==========================================
# 4. 主畫面 UI 呈現 (指標卡片與 Plotly 圖表)
# ==========================================
# 頂部 KPI 卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("估計每月房貸", f"{月繳房貸:.1f} 萬" if 月繳房貸 > 0 else "0 萬")
with col2:
    val_65_invest_house = df.loc[65, '有投資_有買房'] if 65 in df.index else 0
    st.metric("65歲資產 (投資+買房)", f"{val_65_invest_house:.0f} 萬")
with col3:
    val_65_invest_nohouse = df.loc[65, '有投資_無買房'] if 65 in df.index else 0
    st.metric("65歲資產 (僅投資)", f"{val_65_invest_nohouse:.0f} 萬")
with col4:
    st.metric("總投入頭期款", f"{買房頭期款:.0f} 萬")

st.markdown("<br>", unsafe_allow_html=True)

# 繪製 Plotly 折線圖
fig = px.line(
    df, 
    x=df.index, 
    y=['無投資_無買房', '有投資_無買房', '無投資_有買房', '有投資_有買房'],
    labels={'value': '總資產 (萬元)', 'variable': '情境'},
    color_discrete_sequence=['#8E8E93', '#007AFF', '#FF3B30', '#34C759'] # iOS 風格配色
)

fig.update_layout(
    title_text="人生資產變化曲線比較",
    title_font=dict(size=20, family="sans-serif"),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='總資產 (萬元)')
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("📝 財務洞察結論 (點擊展開)"):
    st.write("""
    1. **投資報酬率的雪球效應**：觀察「有投資」與「無投資」的曲線，在後期會因為年化報酬率產生巨大的分岔。
    2. **買房的流動性陷阱**：在買房初期，資產曲線通常會往下掉（因為支付頭期款與高額房貸），但長遠來看，這是一種強迫儲蓄，可與無投資的純租房族拉開差距。
    3. **投資比例的關鍵**：即使買了房，剩餘的可支配現金是否持續投入市場，決定了退休後資產成長的天花板。
    """)