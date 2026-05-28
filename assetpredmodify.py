import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. 核心量化模型：資產負債與蒙地卡羅引擎
# ==========================================
class LifeFinancialALM:
    def __init__(self, params):
        self.p = params
        self.N_years = 101 - self.p['起始年齡']
        self.ages = np.arange(self.p['起始年齡'], 101)
        self.N_paths = self.p.get('模擬路徑數', 1000)

    def generate_base_cashflows(self):
        """產生基礎通膨與現金流 (純向量化計算)"""
        inflation_mult = (1 + self.p['通膨率']) ** np.arange(self.N_years)
        salary = np.where(self.ages <= self.p['退休年齡'], self.p['月薪'] * 12 * inflation_mult, 0)
        expenses = self.p['月開銷'] * 12 * inflation_mult
        rent = self.p['月房租'] * 12 * inflation_mult
        
        net_cashflow_renting = salary - expenses - rent
        return net_cashflow_renting, rent, inflation_mult

    def generate_mortgage_schedule(self):
        """計算房貸攤還 (支援寬限期精算邏輯)"""
        loan_amount = self.p['房價'] - self.p['頭期款']
        annual_rate = self.p['房貸利率']
        total_years = self.p['房貸年限']
        grace_years = self.p['寬限期']
        
        # 例外防呆
        if loan_amount <= 0 or total_years <= 0:
            return np.zeros(self.N_years), np.zeros(self.N_years), 0
            
        yearly_pmt = np.zeros(self.N_years)
        remaining_principal = np.zeros(self.N_years)
        current_principal = loan_amount
        
        # 精算每月利率與期數
        monthly_rate = annual_rate / 12
        
        for y in range(total_years):
            if y < self.N_years:  # 確保不超出生命表邊界
                if y < grace_years:
                    # 寬限期：僅繳利息
                    pmt_year = current_principal * annual_rate
                    yearly_pmt[y] = pmt_year
                    remaining_principal[y] = current_principal
                else:
                    # 攤還期：本息平均攤還
                    rem_months = (total_years - grace_years) * 12
                    monthly_pmt = current_principal * (monthly_rate * (1 + monthly_rate)**rem_months) / ((1 + monthly_rate)**rem_months - 1)
                    
                    # 計算該年年底的剩餘本金 (扣除12個月的本金攤還)
                    for m in range(12):
                        interest = current_principal * monthly_rate
                        principal_paid = monthly_pmt - interest
                        current_principal -= principal_paid
                        
                    yearly_pmt[y] = monthly_pmt * 12
                    remaining_principal[y] = max(current_principal, 0) # 避免浮點數誤差
                    
        return yearly_pmt, remaining_principal, loan_amount

    def build_scenarios(self):
        """建構純租房與買房情境的現金流與非流動資產"""
        cf_renting, rent_array, _ = self.generate_base_cashflows()
        cf_buying = np.copy(cf_renting)
        
        property_value = np.zeros(self.N_years)
        mortgage_balance = np.zeros(self.N_years)
        
        yearly_mortgage_pmt, remaining_principal, loan_amount = self.generate_mortgage_schedule()

        if self.p['買房年齡'] in self.ages:
            idx_buy = np.where(self.ages == self.p['買房年齡'])[0][0]
            idx_end = min(idx_buy + self.p['房貸年限'], self.N_years)
            
            # 買房現金流重構：扣頭期款、停付房租、繳房貸
            cf_buying[idx_buy] -= self.p['頭期款']
            cf_buying[idx_buy:] += rent_array[idx_buy:] 
            
            # 動態匹配房貸繳納陣列長度
            loan_length = idx_end - idx_buy
            cf_buying[idx_buy:idx_end] -= yearly_mortgage_pmt[:loan_length]
            
            # 房產市值 (幾何增值)
            years_owned = np.arange(self.N_years) - idx_buy
            appreciation = (1 + self.p['房產年增值']) ** np.clip(years_owned, 0, None)
            property_value[idx_buy:] = self.p['房價'] * appreciation[idx_buy:]
            
            # 房貸餘額對齊
            mortgage_balance[idx_buy:idx_end] = remaining_principal[:loan_length]

        return cf_renting, cf_buying, property_value, mortgage_balance, yearly_mortgage_pmt

    def simulate_wealth_mc(self, cashflows, is_investing=True):
        """量化蒙地卡羅投資模擬 (處理破產路徑依賴)"""
        wealth = np.zeros((self.N_years, self.N_paths))
        wealth[0, :] = self.p['起始資金'] + cashflows[0]
        
        if is_investing:
            # 預先生成隨機變數以優化效能
            Z = np.random.normal(0, 1, (self.N_years, self.N_paths))
            mu = self.p['預期報酬'] * self.p['投資比例']
            sigma = self.p['預期波動率'] * self.p['投資比例']
            # 使用幾何布朗運動 (Geometric Brownian Motion) 的離散近似
            portfolio_returns = mu - (sigma**2)/2 + sigma * Z
        else:
            portfolio_returns = np.zeros((self.N_years, self.N_paths))
        
        for t in range(1, self.N_years):
            cf = cashflows[t]
            prev_wealth = wealth[t-1, :]
            
            # 破產截斷條件：若流動資產>0則享受市場複利，否則僅累加現金流
            wealth[t, :] = np.where(
                prev_wealth > 0,
                prev_wealth * (1 + portfolio_returns[t, :]) + cf,
                prev_wealth + cf
            )
        return wealth

    def run(self):
        """執行完整模型並回傳統計結果與風險指標"""
        cf_rent, cf_buy, prop_val, mort_bal, mort_pmt = self.build_scenarios()
        
        # 1. 執行蒙地卡羅模擬 (有投資)
        mc_rent_invest = self.simulate_wealth_mc(cf_rent, is_investing=True)
        mc_buy_invest = self.simulate_wealth_mc(cf_buy, is_investing=True)
        
        # 2. 執行確定性計算 (無投資)
        det_rent_no_invest = self.simulate_wealth_mc(cf_rent, is_investing=False)[:, 0]
        det_buy_no_invest = self.simulate_wealth_mc(cf_buy, is_investing=False)[:, 0]
        
        # 計算買房情境的「總淨資產」路徑
        mc_buy_net_worth = mc_buy_invest + prop_val[:, None] - mort_bal[:, None]
        
        # 彙整總淨資產結果
        results = pd.DataFrame(index=self.ages)
        results['年齡'] = self.ages
        results['純租_現金流'] = cf_rent
        results['買房_現金流'] = cf_buy
        
        results['純租_無投資'] = det_rent_no_invest
        results['買房_無投資'] = det_buy_no_invest + prop_val - mort_bal
        
        results['純租投資_中位數'] = np.median(mc_rent_invest, axis=1)
        results['純租投資_P5'] = np.percentile(mc_rent_invest, 5, axis=1)
        results['純租投資_P95'] = np.percentile(mc_rent_invest, 95, axis=1)
        
        results['買房投資_中位數'] = np.median(mc_buy_net_worth, axis=1)
        results['買房投資_P5'] = np.percentile(mc_buy_net_worth, 5, axis=1)
        results['買房投資_P95'] = np.percentile(mc_buy_net_worth, 95, axis=1)
        
        # 3. 風險統計指標 (路徑依賴的破產機率與最大回撤)
        target_age = 65 if 65 in self.ages else self.ages[-1]
        idx_target = np.where(self.ages == target_age)[0][0]
        
        # [修正] 嚴謹的路徑依賴破產判定：只要在目標年齡前，流動資產曾 < 0 即視為破產
        ruin_probs = {
            'buy_inv_65': np.mean(np.any(mc_buy_invest[:idx_target+1, :] < 0, axis=0)) * 100,
            'rent_inv_65': np.mean(np.any(mc_rent_invest[:idx_target+1, :] < 0, axis=0)) * 100,
            'buy_noinv_65': 100.0 if np.any(det_buy_no_invest[:idx_target+1] < 0) else 0.0,
            'rent_noinv_65': 100.0 if np.any(det_rent_no_invest[:idx_target+1] < 0) else 0.0,
            'rent_inv_end': np.mean(np.any(mc_rent_invest < 0, axis=0)) * 100,
            'buy_inv_end': np.mean(np.any(mc_buy_invest < 0, axis=0)) * 100
        }
        
        # [新增] 計算 65 歲前總淨資產的最大回撤率 (Max Drawdown, MDD) 中位數
        def calc_median_mdd(net_worth_paths, end_idx):
            paths_subset = net_worth_paths[:end_idx+1, :]
            peaks = np.maximum.accumulate(paths_subset, axis=0)
            # 避免除以零
            drawdowns = (paths_subset - peaks) / np.where(peaks <= 0, 1e-9, peaks)
            # 將因淨資產為負造成的異常值截斷
            drawdowns = np.clip(drawdowns, -1, 0)
            max_drawdowns = np.min(drawdowns, axis=0)
            return abs(np.median(max_drawdowns)) * 100

        mdd_stats = {
            'buy_mdd': calc_median_mdd(mc_buy_net_worth, idx_target),
            'rent_mdd': calc_median_mdd(mc_rent_invest, idx_target)
        }

        return results, mort_pmt, ruin_probs, mdd_stats

# ==========================================
# 2. 頁面設定與 UI 渲染
# ==========================================
st.set_page_config(page_title="量化人生財務模擬器", page_icon="📈", layout="wide")
st.title("📈 量化人生財務模擬器 (Monte Carlo Edition)")
st.markdown("融合**精算科學**與**蒙地卡羅模擬**，真實呈現純儲蓄與市場投資在不同決策下的量化邊界。")

with st.sidebar:
    st.header("⚙️ 量化參數設定")
    
    st.subheader("👤 基本財務")
    p_age = st.number_input("目前年齡", min_value=20, max_value=60, value=30, step=1)
    p_retire = st.number_input("預計退休年齡", min_value=40, max_value=80, value=65, step=1)
    p_capital = st.number_input("起始流動資金 (萬元)", min_value=0, value=200, step=10)
    p_inflation = st.number_input("年通膨/薪資成長率 (%)", min_value=0.0, value=2.0, step=0.5) / 100.0
    
    p_salary = st.number_input("目前平均月薪 (萬元)", min_value=0.0, value=8.0, step=0.5)
    p_expense = st.number_input("目前月開銷 (萬元)", min_value=0.0, value=3.0, step=0.1)
    p_rent = st.number_input("目前月房租 (萬元)", min_value=0.0, value=2.0, step=0.1)
    
    st.markdown("---")
    st.subheader("📊 投資市場動態")
    p_inv_ratio = st.slider("可支配資金投資比例 (%)", 0, 100, 70, 5) / 100.0
    p_return = st.number_input("預期年化報酬率 (μ) (%)", value=7.0, step=0.5) / 100.0
    p_volatility = st.number_input("年化波動率 (σ) (%)", value=15.0, step=0.5) / 100.0
    p_paths = st.selectbox("蒙地卡羅路徑數", options=[100, 500, 1000, 5000], index=2)
    
    st.markdown("---")
    st.subheader("🏠 購屋與貸款模組 (Mortgage Input)")
    p_buy_age = st.number_input("預計買房年齡", min_value=p_age, max_value=80, value=35, step=1)
    p_house_price = st.number_input("房屋總價 (萬元)", min_value=100, value=1500, step=100)
    p_down_pmt = st.number_input("頭期款 (萬元)", min_value=100, value=300, step=50)
    
    col_a, col_b = st.columns(2)
    with col_a:
        p_loan_years = st.number_input("房貸年限", value=30, step=1)
    with col_b:
        p_grace_years = st.number_input("寬限期", min_value=0, max_value=10, value=3, step=1)
        
    p_loan_rate = st.number_input("房貸利率 (%)", value=2.1, step=0.1) / 100.0
    p_house_appr = st.number_input("房產年增值率 (%)", value=1.5, step=0.1) / 100.0

params = {
    '起始年齡': p_age, '退休年齡': p_retire, '起始資金': p_capital, '通膨率': p_inflation,
    '月薪': p_salary, '月開銷': p_expense, '月房租': p_rent,
    '投資比例': p_inv_ratio, '預期報酬': p_return, '預期波動率': p_volatility, '模擬路徑數': p_paths,
    '買房年齡': p_buy_age, '房價': p_house_price, '頭期款': p_down_pmt,
    '房產年增值': p_house_appr, '房貸年限': p_loan_years, '寬限期': p_grace_years, '房貸利率': p_loan_rate
}

model = LifeFinancialALM(params)
df_res, mort_pmt, ruin_probs, mdd_stats = model.run()

# ==========================================
# 3. 儀表板與數據視覺化
# ==========================================
target_age = 65 if 65 in df_res.index else df_res.index[-1]
target_data = df_res.loc[target_age]

st.subheader(f"📊 {target_age}歲 財務健康度總覽 (淨資產與破產風險)")

# --- 第一排：65歲 淨資產 (中位數) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(f"{target_age}歲淨資產 (買房+投資)", f"{target_data['買房投資_中位數']:.0f} 萬")
with col2:
    st.metric(f"{target_age}歲淨資產 (純租+投資)", f"{target_data['純租投資_中位數']:.0f} 萬")
with col3:
    st.metric(f"{target_age}歲淨資產 (買房純儲蓄)", f"{target_data['買房_無投資']:.0f} 萬")
with col4:
    st.metric(f"{target_age}歲淨資產 (純租純儲蓄)", f"{target_data['純租_無投資']:.0f} 萬")

st.markdown("<br>", unsafe_allow_html=True)

# --- 第二排：65歲 累積破產率 (Path-Dependent Ruin Prob) ---
col5, col6, col7, col8 = st.columns(4)
with col5:
    st.metric(f"{target_age}歲前 (買房+投資) 破產率", f"{ruin_probs['buy_inv_65']:.1f}%", 
              delta=f"最大回撤: -{mdd_stats['buy_mdd']:.1f}%", delta_color="inverse")
with col6:
    st.metric(f"{target_age}歲前 (純租+投資) 破產率", f"{ruin_probs['rent_inv_65']:.1f}%",
              delta=f"最大回撤: -{mdd_stats['rent_mdd']:.1f}%", delta_color="inverse")
with col7:
    st.metric(f"{target_age}歲前 (買房純儲蓄) 破產率", f"{ruin_probs['buy_noinv_65']:.1f}%")
with col8:
    st.metric(f"{target_age}歲前 (純租純儲蓄) 破產率", f"{ruin_probs['rent_noinv_65']:.1f}%")

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- 圖表 1：蒙地卡羅淨資產曲線 -----------------
st.subheader("💰 1. 人生真實淨資產 (Net Worth) 蒙地卡羅預測")
st.markdown("實線為**純儲蓄（無投資）**之結果；虛線為投資**中位數**，陰影區域代表投資的 **90% 信心區間**。")

fig_nw = go.Figure()

# 買房 + 投資
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_P95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_P5'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 199, 89, 0.2)', name='買房+投資 (90% 信心區間)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_中位數'], mode='lines', line=dict(color='#34C759', width=3, dash='dash'), name='買房+投資 (中位數)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房_無投資'], mode='lines', line=dict(color='#FF3B30', width=3), name='買房純儲蓄'))

# 純租 + 投資
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_P95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_P5'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 122, 255, 0.2)', name='純租+投資 (90% 信心區間)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_中位數'], mode='lines', line=dict(color='#007AFF', width=3, dash='dash'), name='純租+投資 (中位數)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租_無投資'], mode='lines', line=dict(color='#8E8E93', width=3), name='純租純儲蓄'))

fig_nw.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='真實淨資產 (萬元)')
)
st.plotly_chart(fig_nw, use_container_width=True)

# ----------------- 財務洞察 -----------------
with st.expander("📝 專家量化洞察報告 (點擊展開)", expanded=True):
    st.write(f"""
    ### 深度量化診斷結果
    
    | 診斷維度 | 量化回測事實 | 財務意義 |
    | :--- | :--- | :--- |
    | **破產風險評估 <br> (Probability of Ruin)** | 終老純租破產率：**{ruin_probs['rent_inv_end']:.1f}%** <br> 終老買房破產率：**{ruin_probs['buy_inv_end']:.1f}%** | 修正為**路徑依賴演算法**後，只要在生命週期中「曾跌破 0 萬元」即計入破產。買房初期的頭期款加上寬限期結束後的本息攤還，極易在市場下行時觸發流動性違約危機。 |
    | **最大回撤與波動 <br> (Max Drawdown)** | 買房投資 MDD：**-{mdd_stats['buy_mdd']:.1f}%** <br> 純租投資 MDD：**-{mdd_stats['rent_mdd']:.1f}%** | 純租情境的資產完全暴露於市場波動；買房情境則因房產本身的低波動性（年增值 {p_house_appr*100:.1f}%）提供了實質的淨資產下檔保護。 |
    | **寬限期與槓桿 <br> (Grace Period & Leverage)** | 房貸寬限期：**{p_grace_years} 年** <br> 寬限期滿月繳：**{(mort_pmt[p_grace_years]/12):.1f} 萬** | 寬限期透過延遲本金償還，將資金釋放至市場賺取風險溢酬（Risk Premium）。本質上是利用定息債務作為做空法定貨幣購買力的對沖工具。 |
    """)