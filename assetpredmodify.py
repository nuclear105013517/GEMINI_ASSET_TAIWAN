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
        """產生基礎通膨與現金流 (向量化)"""
        inflation_mult = (1 + self.p['通膨率']) ** np.arange(self.N_years)
        salary = np.where(self.ages <= self.p['退休年齡'], self.p['月薪'] * 12 * inflation_mult, 0)
        expenses = self.p['月開銷'] * 12 * inflation_mult
        rent = self.p['月房租'] * 12 * inflation_mult
        
        net_cashflow_renting = salary - expenses - rent
        return net_cashflow_renting, rent, inflation_mult

    def generate_mortgage_schedule(self):
        """計算房貸攤還與房屋市值變化 (精算邏輯)"""
        loan_amount = self.p['房價'] - self.p['頭期款']
        monthly_rate = self.p['房貸利率'] / 12
        total_months = self.p['房貸年限'] * 12
        
        if loan_amount <= 0 or total_months <= 0:
            return 0, np.zeros(self.N_years), np.zeros(self.N_years)
            
        monthly_pmt = loan_amount * (monthly_rate * (1 + monthly_rate)**total_months) / ((1 + monthly_rate)**total_months - 1)
        yearly_pmt = monthly_pmt * 12
        
        months_passed = np.arange(0, self.N_years * 12 + 1, 12)
        months_passed = np.clip(months_passed, 0, total_months)
        
        factor_n = (1 + monthly_rate) ** total_months
        factor_k = (1 + monthly_rate) ** months_passed
        remaining_principal = loan_amount * (factor_n - factor_k) / (factor_n - 1)
        
        return yearly_pmt, remaining_principal, loan_amount

    def build_scenarios(self):
        """建構純租房與買房情境的現金流與非流動資產"""
        cf_renting, rent_array, inflation_mult = self.generate_base_cashflows()
        cf_buying = np.copy(cf_renting)
        
        property_value = np.zeros(self.N_years)
        mortgage_balance = np.zeros(self.N_years)
        yearly_mortgage_pmt, remaining_principal, loan_amount = self.generate_mortgage_schedule()

        if self.p['買房年齡'] in self.ages:
            idx_buy = np.where(self.ages == self.p['買房年齡'])[0][0]
            idx_end = min(idx_buy + self.p['房貸年限'], self.N_years)
            
            cf_buying[idx_buy] -= self.p['頭期款']
            cf_buying[idx_buy:] += rent_array[idx_buy:] 
            cf_buying[idx_buy:idx_end] -= yearly_mortgage_pmt
            
            years_owned = np.arange(self.N_years) - idx_buy
            appreciation = (1 + self.p['房產年增值']) ** np.clip(years_owned, 0, None)
            property_value[idx_buy:] = self.p['房價'] * appreciation[idx_buy:]
            
            mortgage_balance[idx_buy:idx_end] = remaining_principal[1 : idx_end - idx_buy + 1]

        return cf_renting, cf_buying, property_value, mortgage_balance, yearly_mortgage_pmt

    def simulate_wealth_mc(self, cashflows, is_investing=True):
        """向量化蒙地卡羅投資模擬 (處理破產路徑依賴與零投資情境)"""
        wealth = np.zeros((self.N_years, self.N_paths))
        wealth[0, :] = self.p['起始資金'] + cashflows[0]
        
        if is_investing:
            Z = np.random.normal(0, 1, (self.N_years, self.N_paths))
            mu = self.p['預期報酬'] * self.p['投資比例']
            sigma = self.p['預期波動率'] * self.p['投資比例']
            portfolio_returns = mu - (sigma**2)/2 + sigma * Z
        else:
            # 不投資時，報酬率為 0 (無波動，無增長)
            portfolio_returns = np.zeros((self.N_years, self.N_paths))
        
        for t in range(1, self.N_years):
            cf = cashflows[t]
            prev_wealth = wealth[t-1, :]
            
            wealth[t, :] = np.where(
                prev_wealth > 0,
                prev_wealth * (1 + portfolio_returns[t, :]) + cf,
                prev_wealth + cf
            )
        return wealth

    def run(self):
        """執行完整模型並回傳統計結果"""
        cf_rent, cf_buy, prop_val, mort_bal, mort_pmt = self.build_scenarios()
        
        # 1. 執行蒙地卡羅模擬 (有投資)
        mc_rent_invest = self.simulate_wealth_mc(cf_rent, is_investing=True)
        mc_buy_invest = self.simulate_wealth_mc(cf_buy, is_investing=True)
        
        # 2. 執行確定性計算 (無投資)
        det_rent_no_invest = self.simulate_wealth_mc(cf_rent, is_investing=False)[:, 0]
        det_buy_no_invest = self.simulate_wealth_mc(cf_buy, is_investing=False)[:, 0]
        
        # 彙整結果
        results = pd.DataFrame(index=self.ages)
        results['年齡'] = self.ages
        results['純租_現金流'] = cf_rent
        results['買房_現金流'] = cf_buy
        
        # 紀錄無投資情境的真實淨資產
        results['純租_無投資'] = det_rent_no_invest
        results['買房_無投資'] = det_buy_no_invest + prop_val - mort_bal
        
        # 紀錄有投資情境的真實淨資產 (中位數與信心區間)
        results['純租投資_中位數'] = np.median(mc_rent_invest, axis=1)
        results['純租投資_P5'] = np.percentile(mc_rent_invest, 5, axis=1)
        results['純租投資_P95'] = np.percentile(mc_rent_invest, 95, axis=1)
        
        mc_buy_net_worth = mc_buy_invest + prop_val[:, None] - mort_bal[:, None]
        results['買房投資_中位數'] = np.median(mc_buy_net_worth, axis=1)
        results['買房投資_P5'] = np.percentile(mc_buy_net_worth, 5, axis=1)
        results['買房投資_P95'] = np.percentile(mc_buy_net_worth, 95, axis=1)
        
        # 破產機率統計 (終老時流動資產小於0的機率)
        ruin_prob_rent = np.mean(mc_rent_invest[-1, :] < 0) * 100
        ruin_prob_buy = np.mean(mc_buy_invest[-1, :] < 0) * 100
        
        return results, mort_pmt, ruin_prob_rent, ruin_prob_buy

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
    st.subheader("📊 投資市場動態 (Market Dynamics)")
    p_inv_ratio = st.slider("可支配資金投資比例 (%)", 0, 100, 70, 5) / 100.0
    p_return = st.number_input("預期年化報酬率 (μ) (%)", value=7.0, step=0.5) / 100.0
    p_volatility = st.number_input("年化波動率 (σ) (%)", value=15.0, step=0.5) / 100.0
    p_paths = st.selectbox("蒙地卡羅路徑數", [100, 500, 1000, 5000], index=1)
    
    st.markdown("---")
    st.subheader("🏠 購屋資產互換")
    p_buy_age = st.number_input("預計買房年齡", min_value=p_age, max_value=80, value=35, step=1)
    p_house_price = st.number_input("房屋總價 (萬元)", min_value=100, value=1500, step=100)
    p_down_pmt = st.number_input("頭期款 (萬元)", min_value=100, value=300, step=50)
    p_house_appr = st.number_input("房產年增值率 (%)", value=1.5, step=0.1) / 100.0
    p_loan_years = st.number_input("房貸年限 (年)", value=30, step=1)
    p_loan_rate = st.number_input("房貸利率 (%)", value=2.1, step=0.1) / 100.0

params = {
    '起始年齡': p_age, '退休年齡': p_retire, '起始資金': p_capital, '通膨率': p_inflation,
    '月薪': p_salary, '月開銷': p_expense, '月房租': p_rent,
    '投資比例': p_inv_ratio, '預期報酬': p_return, '預期波動率': p_volatility, '模擬路徑數': p_paths,
    '買房年齡': p_buy_age, '房價': p_house_price, '頭期款': p_down_pmt,
    '房產年增值': p_house_appr, '房貸年限': p_loan_years, '房貸利率': p_loan_rate
}

model = LifeFinancialALM(params)
df_res, mort_pmt, ruin_rent, ruin_buy = model.run()

# ==========================================
# 3. 儀表板與數據視覺化
# ==========================================
col1, col2, col3, col4 = st.columns(4)
target_age = 65 if 65 in df_res.index else df_res.index[-1]
target_data = df_res.loc[target_age]

with col1:
    st.metric(f"65歲淨資產 (買房+投資)", f"{target_data['買房投資_中位數']:.0f} 萬")
with col2:
    st.metric(f"65歲淨資產 (純租+投資)", f"{target_data['純租投資_中位數']:.0f} 萬")
with col3:
    st.metric(f"65歲淨資產 (買房純儲蓄)", f"{target_data['買房_無投資']:.0f} 萬")
with col4:
    st.metric(f"65歲淨資產 (純租純儲蓄)", f"{target_data['純租_無投資']:.0f} 萬")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- 圖表 1：蒙地卡羅淨資產曲線 -----------------
st.subheader("💰 1. 人生真實淨資產 (Net Worth) 蒙地卡羅預測與純儲蓄對比")
st.markdown("實線為**純儲蓄（無投資）**之結果；虛線為投資**中位數**，陰影區域代表投資的 **90% 信心區間**。")

fig_nw = go.Figure()

# --- 買房情境群 (綠色/紅色系) ---
# 買房 + 投資 (綠色系)
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_P95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_P5'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 199, 89, 0.2)', name='買房+投資 (90% 信心區間)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_中位數'], mode='lines', line=dict(color='#34C759', width=3, dash='dash'), name='買房+投資 (中位數)'))

# 買房無投資 (紅色實線)
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房_無投資'], mode='lines', line=dict(color='#FF3B30', width=3), name='買房+純儲蓄'))

# --- 純租情境群 (藍色/灰色系) ---
# 純租 + 投資 (藍色系)
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_P95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_P5'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 122, 255, 0.2)', name='純租+投資 (90% 信心區間)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_中位數'], mode='lines', line=dict(color='#007AFF', width=3, dash='dash'), name='純租+投資 (中位數)'))

# 純租無投資 (灰色實線)
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租_無投資'], mode='lines', line=dict(color='#8E8E93', width=3), name='純租+純儲蓄'))


fig_nw.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='真實淨資產 (萬元)')
)
st.plotly_chart(fig_nw, use_container_width=True)

# ----------------- 圖表 2：年度收支平衡曲線 -----------------
st.subheader("⚖️ 2. 年度淨現金流曲線 (Cash Flow Dynamics)")
fig_cf = px.line(
    df_res, x='年齡', y=['純租_現金流', '買房_現金流'],
    labels={'value': '年度淨現金流 (萬元)', 'variable': '現金流情境'},
    color_discrete_sequence=['#007AFF', '#FF3B30']
)
fig_cf.add_hline(y=0, line_dash="dash", line_color="#8E8E93", annotation_text="收支平衡線 (0 萬元)", annotation_position="bottom right")
fig_cf.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年度淨現金流 (萬元)')
)
st.plotly_chart(fig_cf, use_container_width=True)