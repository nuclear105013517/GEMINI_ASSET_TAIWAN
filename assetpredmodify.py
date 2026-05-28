import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==========================================
# 1. 頁面設定與 iOS 風格 UI 初始化
# ==========================================
st.set_page_config(page_title="人生財務曲線預測", page_icon="📈", layout="wide")

st.title("📈 預測你的人生財務曲線")
st.markdown("透過嚴謹的資產負債表（Balance Sheet）模型，真實視覺化**純儲蓄**、**資產投資**與**購屋決策**對未來**真實淨資產 (Net Worth)**與**年度現金流 (Cash Flow)**的影響。")

# ==========================================
# 2. 側邊欄參數設定 (Sidebar Inputs)
# ==========================================
with st.sidebar:
    st.header("⚙️ 參數設定")
    
    st.subheader("👤 基本財務與總經環境")
    起始年齡 = st.number_input("目前年齡", min_value=20, max_value=60, value=32, step=1)
    退休年齡 = st.number_input("預計退休年齡", min_value=40, max_value=80, value=65, step=1)
    起始資金 = st.number_input("起始流動資金 (萬元)", min_value=0, value=100, step=10)
    通膨與薪資成長 = st.number_input("預估年通膨/薪資成長率 (%)", min_value=0.0, value=2.0, step=0.5) / 100.0
    
    每月薪水 = st.number_input("目前平均每月薪水 (萬元)", min_value=0.0, value=8.0, step=0.5)
    每月開銷 = st.number_input("目前每月生活開銷 (萬元)", min_value=0.0, value=3.0, step=0.1, help="不含房租與房貸")
    每月房租 = st.number_input("目前每月房租 (萬元)", min_value=0.0, value=2.0, step=0.1)
    
    st.markdown("---")
    st.subheader("📈 投資策略")
    投資部位 = st.slider("可支配資金投資比例 (%)", min_value=0, max_value=100, value=70, step=5) / 100.0
    投資年利率 = st.number_input("預期年化報酬率 (%)", min_value=0.0, value=6.0, step=0.5) / 100.0
    
    st.markdown("---")
    st.subheader("🏠 購屋計畫 (資產互換模型)")
    買房年齡 = st.number_input("預計買房年齡", min_value=起始年齡, max_value=80, value=40, step=1)
    買房價格 = st.number_input("房屋總價 (萬元)", min_value=100, value=1500, step=100)
    買房頭期款 = st.number_input("頭期款 (萬元)", min_value=100, value=300, step=50)
    房產年增值 = st.number_input("預期房產年增值率 (%)", min_value=0.0, value=1.5, step=0.1) / 100.0
    貸款年數 = st.number_input("房貸年限 (年)", min_value=10, max_value=40, value=30, step=1)
    房貸利率 = st.number_input("房貸利率 (%)", min_value=1.0, value=2.1, step=0.1) / 100.0

# ==========================================
# 3. 核心財務計算邏輯 (資產負債表與現金流視角)
# ==========================================
預測時段 = np.arange(起始年齡, 101)
df = pd.DataFrame(index=預測時段)
df.index.name = '年齡'
N_years = len(預測時段)

# 考慮通膨的指數成長係數
inflation_multiplier = (1 + 通膨與薪資成長) ** np.arange(N_years)

# 計算年度基礎現金流 (動態調整通膨)
年薪 = np.where(df.index <= 退休年齡, 每月薪水 * 12 * inflation_multiplier, 0)
年開銷 = 每月開銷 * 12 * inflation_multiplier
年房租 = 每月房租 * 12 * inflation_multiplier
基礎年淨額 = 年薪 - 年開銷 - 年房租

# --- 房貸本息攤還與剩餘本金試算 (全向量化) ---
月利率 = 房貸利率 / 12
貸款總額 = 買房價格 - 買房頭期款
總月數 = 貸款年數 * 12

if 貸款總額 > 0 and 貸款年數 > 0:
    月繳房貸 = 貸款總額 * (月利率 * (1 + 月利率)**總月數) / ((1 + 月利率)**總月數 - 1)
    年繳房貸 = 月繳房貸 * 12
    
    months_passed = np.arange(0, N_years * 12 + 1, 12)
    months_passed = np.clip(months_passed, 0, 總月數)
    
    factor_n = (1 + 月利率) ** 總月數
    factor_k = (1 + 月利率) ** months_passed
    剩餘房貸陣列 = 貸款總額 * (factor_n - factor_k) / (factor_n - 1)
else:
    月繳房貸 = 0
    年繳房貸 = 0
    剩餘房貸陣列 = np.zeros(N_years + 1)

# --- 建構買房情境的現金流與非流動資產 ---
買房年淨額 = np.copy(基礎年淨額)
房產市值陣列 = np.zeros(N_years)
年度剩餘房貸 = np.zeros(N_years)

if 買房年齡 in df.index:
    idx_buy = np.where(df.index == 買房年齡)[0][0]
    
    # 現金流調整 (扣除頭期款、停止付租、開始付房貸)
    買房年淨額[idx_buy] -= 買房頭期款
    idx_mortgage_end = min(idx_buy + 貸款年數, N_years)
    買房年淨額[idx_buy:] += 年房租[idx_buy:] 
    買房年淨額[idx_buy:idx_mortgage_end] -= 年繳房貸
    
    # 非流動資產價值計算
    years_since_buy = np.arange(N_years) - idx_buy
    房產增值係數 = (1 + 房產年增值) ** np.clip(years_since_buy, 0, None)
    房產市值陣列[idx_buy:] = 買房價格 * 房產增值係數[idx_buy:]
    
    年度剩餘房貸[idx_buy:idx_mortgage_end] = 剩餘房貸陣列[1 : idx_mortgage_end - idx_buy + 1]

# --- 紀錄年度收支平衡數據 ---
df['純租房_年度淨現金流'] = 基礎年淨額
df['有買房_年度淨現金流'] = 買房年淨額

# --- 複利計算函式 (支援破產截斷條件) ---
def calculate_liquid_wealth(net_cash_flow, start_cap, inv_ratio, inv_rate):
    wealth = np.zeros(len(net_cash_flow))
    current = float(start_cap)
    alpha = 1.0 + (inv_rate * inv_ratio)
    
    for i in range(len(net_cash_flow)):
        cf = net_cash_flow[i]
        if current > 0:
            current = current * alpha + cf
        else:
            current = current + cf
        wealth[i] = current
    return wealth

# 產出流動資產數據
流動_無投資_無買房 = calculate_liquid_wealth(基礎年淨額, 起始資金, 0, 0)
流動_有投資_無買房 = calculate_liquid_wealth(基礎年淨額, 起始資金, 投資部位, 投資年利率)
流動_無投資_有買房 = calculate_liquid_wealth(買房年淨額, 起始資金, 0, 0)
流動_有投資_有買房 = calculate_liquid_wealth(買房年淨額, 起始資金, 投資部位, 投資年利率)

# 計算總真實淨資產
df['無投資_無租房(純租)'] = 流動_無投資_無買房
df['有投資_無買房(純租)'] = 流動_有投資_無買房
df['無投資_有買房'] = 流動_無投資_有買房 + 房產市值陣列 - 年度剩餘房貸
df['有投資_有買房'] = 流動_有投資_有買房 + 房產市值陣列 - 年度剩餘房貸

# ==========================================
# 4. 主畫面 UI 呈現 (指標卡片與 Plotly 圖表)
# ==========================================
col1, col2, col3, col4 = st.columns(4)

target_age = 65 if 65 in df.index else df.index[-1]

with col1:
    st.metric("估計每月房貸", f"{月繳房貸:.1f} 萬" if 月繳房貸 > 0 else "0 萬")
with col2:
    val_invest_house = df.loc[target_age, '有投資_有買房']
    st.metric(f"{target_age}歲真實淨資產 (投資+買房)", f"{val_invest_house:.0f} 萬")
with col3:
    val_invest_nohouse = df.loc[target_age, '有投資_無買房(純租)']
    st.metric(f"{target_age}歲真實淨資產 (僅投資)", f"{val_invest_nohouse:.0f} 萬")
with col4:
    st.metric("總投入頭期款", f"{買房頭期款:.0f} 萬")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- 圖表 1：真實淨資產曲線 -----------------
st.subheader("💰 1. 人生真實淨資產 (Net Worth) 變化曲線")
fig_nw = px.line(
    df, 
    x=df.index, 
    y=['無投資_無租房(純租)', '有投資_無買房(純租)', '無投資_有買房', '有投資_有買房'],
    labels={'value': '真實淨資產 (萬元)', 'variable': '財務情境'},
    color_discrete_sequence=['#8E8E93', '#007AFF', '#FF3B30', '#34C759']
)
fig_nw.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='真實淨資產 (萬元)')
)
st.plotly_chart(fig_nw, use_container_width=True)

# ----------------- 圖表 2：年度收支平衡曲線 -----------------
st.subheader("⚖️ 2. 年度收支平衡曲線 (Cash Flow)")
st.markdown("觀察買房當年的**現金流缺口（頭期款）**，以及退休後無薪資收入的**現金消耗速率**。")
fig_cf = px.line(
    df, 
    x=df.index, 
    y=['純租房_年度淨現金流', '有買房_年度淨現金流'],
    labels={'value': '年度淨現金流 (萬元)', 'variable': '現金流情境'},
    color_discrete_sequence=['#007AFF', '#FF3B30']
)
# 新增收支平衡零軸線 (Zero-line)
fig_cf.add_hline(y=0, line_dash="dash", line_color="#8E8E93", annotation_text="收支平衡線 (0 萬元)", annotation_position="bottom right")

fig_cf.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年度淨現金流 (萬元)')
)
st.plotly_chart(fig_cf, use_container_width=True)

# ----------------- 財務洞察 -----------------
with st.expander("📝 深度量化財務洞察 (點擊展開)"):
    st.write("""
    1. **流動性陷阱的視覺化 (Liquidity Trap)**：在第二張圖（收支平衡曲線）中，你可以明顯看到買房當年度會有一個巨大的向下尖峰，這就是「頭期款」造成的現金流重擊。投資人必須確保「起始資金」能扛住這個衝擊，否則會面臨真實世界的違約破產邊緣。
    2. **房貸與房租的黃金交叉**：觀察收支曲線中，房貸繳清的那一年，買房者的現金流會瞬間大幅反彈向上（因為不再需要付房貸，且本身不用付房租）。相比之下，純租房者終其一生都要承受不斷被通膨墊高的房租壓力。
    3. **退休後的現金流考驗**：當到達預計退休年齡，薪水歸零後，收支曲線將掉落至零軸（收支平衡線）以下。此時每年消耗的金額，將完全取決於你的淨資產（第一張圖）是否厚實到足以產生足夠的被動投資收益來填補。
    """)