import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. 核心量化模型：資產負債與蒙地卡羅引擎 (支援定期定額與動態提領)
# ==========================================
class LifeFinancialALM:
    """
    機構級資產負債管理與蒙地卡羅模擬引擎
    """
    def __init__(self, params):
        self.p = params
        self.N_years = 101 - self.p['起始年齡']
        self.ages = np.arange(self.p['起始年齡'], 101)
        self.N_paths = self.p.get('模擬路徑數', 1000)
        self.payout_annual = 0.0  # 紀錄勞退年領金額

    def generate_base_cashflows(self):
        """產生基礎通膨與現金流，並納入台灣勞退帳戶精算 (純向量化計算)"""
        inflation_mult = (1 + self.p['通膨率']) ** np.arange(self.N_years)
        salary = np.where(self.ages <= self.p['退休年齡'], self.p['月薪'] * 12 * inflation_mult, 0)
        expenses = self.p['月開銷'] * 12 * inflation_mult
        rent = self.p['月房租'] * 12 * inflation_mult
        
        pension_cf_net = np.zeros(self.N_years)
        idx_retire = max(0, self.p['退休年齡'] - self.p['起始年齡']) if self.p['退休年齡'] in self.ages else 0
        
        pension_return = self.p['勞退報酬率']
        current_pension = self.p['勞退目前提撥']
        
        if idx_retire > 0:
            contributions = self.p['勞退每月提撥'] * 12 * inflation_mult[:idx_retire]
            compound_factors = (1 + pension_return) ** np.arange(idx_retire - 1, -1, -1)
            accumulated_contributions = np.sum(contributions * compound_factors)
            current_pension = current_pension * ((1 + pension_return) ** idx_retire) + accumulated_contributions
            
        payout_years = 84 - self.p['退休年齡']
        
        if payout_years > 0 and current_pension > 0:
            if pension_return > 0:
                payout_annual = current_pension * (pension_return * (1 + pension_return)**payout_years) / ((1 + pension_return)**payout_years - 1)
            else:
                payout_annual = current_pension / payout_years
                
            self.payout_annual = payout_annual
            mask_payout = (self.ages >= self.p['退休年齡']) & (self.ages <= 84)
            pension_cf_net[mask_payout] = payout_annual
            
        elif payout_years <= 0 and idx_retire < self.N_years:
            self.payout_annual = current_pension
            pension_cf_net[idx_retire] = current_pension

        net_cashflow_renting = salary - expenses - rent + pension_cf_net
        return net_cashflow_renting, rent, inflation_mult

    def generate_personal_loan_schedule(self):
        balance = self.p['信貸餘額']
        annual_rate = self.p['信貸利率']
        total_years = self.p['信貸年限']
        
        yearly_pmt = np.zeros(self.N_years)
        remaining_principal = np.zeros(self.N_years)
        
        if balance <= 0 or total_years <= 0:
            return yearly_pmt, remaining_principal
            
        monthly_rate = annual_rate / 12
        months = total_years * 12
        
        if monthly_rate > 0:
            monthly_pmt = balance * (monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)
        else:
            monthly_pmt = balance / months
            
        amort_yearly_pmt = monthly_pmt * 12
        valid_years = min(total_years, self.N_years)
        
        if valid_years > 0:
            yearly_pmt[:valid_years] = amort_yearly_pmt
            y_idx = np.arange(1, valid_years + 1)
            months_left = months - (y_idx * 12)
            
            if monthly_rate > 0:
                rem_prin = monthly_pmt * (1 - (1 + monthly_rate)**(-months_left)) / monthly_rate
            else:
                rem_prin = monthly_pmt * months_left
                
            remaining_principal[:valid_years] = np.clip(rem_prin, 0, None)
            
        return yearly_pmt, remaining_principal

    def generate_mortgage_schedule(self):
        loan_amount = self.p['房價'] - self.p['頭期款']
        annual_rate = self.p['房貸利率']
        total_years = self.p['房貸年限']
        grace_years = self.p['寬限期']
        
        yearly_pmt = np.zeros(self.N_years)
        remaining_principal = np.zeros(self.N_years)
        
        if loan_amount <= 0 or total_years <= 0:
            return yearly_pmt, remaining_principal, 0
            
        monthly_rate = annual_rate / 12
        amort_years = total_years - grace_years
        rem_months = amort_years * 12
        
        grace_yearly_pmt = loan_amount * annual_rate
        
        if rem_months > 0:
            monthly_pmt = loan_amount * (monthly_rate * (1 + monthly_rate)**rem_months) / ((1 + monthly_rate)**rem_months - 1)
        else:
            monthly_pmt = 0
            
        amort_yearly_pmt = monthly_pmt * 12
        
        valid_grace = min(grace_years, self.N_years)
        valid_total = min(total_years, self.N_years)
        
        if valid_grace > 0:
            yearly_pmt[:valid_grace] = grace_yearly_pmt
            remaining_principal[:valid_grace] = loan_amount
            
        if valid_total > valid_grace:
            yearly_pmt[valid_grace:valid_total] = amort_yearly_pmt
            y_idx = np.arange(1, valid_total - valid_grace + 1)
            months_left = rem_months - (y_idx * 12)
            
            rem_prin = monthly_pmt * (1 - (1 + monthly_rate)**(-months_left)) / monthly_rate
            remaining_principal[valid_grace:valid_total] = np.clip(rem_prin, 0, None) 

        return yearly_pmt, remaining_principal, loan_amount

    def build_scenarios(self):
        cf_renting, rent_array, _ = self.generate_base_cashflows()
        
        pl_pmt, pl_bal = self.generate_personal_loan_schedule()
        cf_renting -= pl_pmt
        cf_buying = np.copy(cf_renting)
        
        property_value = np.zeros(self.N_years)
        mortgage_balance = np.zeros(self.N_years)
        yearly_mortgage_pmt, remaining_principal, loan_amount = self.generate_mortgage_schedule()

        if self.p['買房年齡'] in self.ages:
            idx_buy = np.where(self.ages == self.p['買房年齡'])[0][0]
            idx_end = min(idx_buy + self.p['房貸年限'], self.N_years)
            loan_length = idx_end - idx_buy
            
            cf_buying[idx_buy] -= self.p['頭期款']
            cf_buying[idx_buy:] += rent_array[idx_buy:] 
            cf_buying[idx_buy:idx_end] -= yearly_mortgage_pmt[:loan_length]
            
            years_owned = np.arange(self.N_years) - idx_buy
            appreciation = (1 + self.p['房產年增值']) ** np.clip(years_owned, 0, None)
            property_value[idx_buy:] = self.p['房價'] * appreciation[idx_buy:]
            mortgage_balance[idx_buy:idx_end] = remaining_principal[:loan_length]

        return cf_renting, cf_buying, property_value, mortgage_balance, yearly_mortgage_pmt, pl_pmt, pl_bal

    def simulate_wealth_mc(self, cashflows, is_investing=True):
        """重新設計：分離投資部位與現金部位，精準模擬 SIP 定期定額與自動變賣生活費邏輯"""
        wealth = np.zeros((self.N_years, self.N_paths))
        inv_wealth = np.zeros((self.N_years, self.N_paths))
        cash_wealth = np.zeros((self.N_years, self.N_paths))
        
        # 初始化第 0 年
        inv_wealth[0, :] = self.p['現有投資']
        cash_wealth[0, :] = self.p['起始資金'] + cashflows[0]
        wealth[0, :] = inv_wealth[0, :] + cash_wealth[0, :]
        
        if is_investing:
            Z = np.random.standard_normal((self.N_years, self.N_paths))
            mu = self.p['預期報酬']
            sigma = self.p['預期波動率']
            M = np.exp((mu - (sigma**2)/2) + sigma * Z)
            annual_inv = self.p['每月投資'] * 12
        else:
            M = np.ones((self.N_years, self.N_paths))
            annual_inv = 0.0
            
        for t in range(1, self.N_years):
            prev_inv = inv_wealth[t-1, :]
            prev_cash = cash_wealth[t-1, :]
            
            # 1. 投資部位隨市場波動，現金部位加上當年度收支
            current_inv = prev_inv * M[t, :]
            current_cash = prev_cash + cashflows[t]
            
            total_w = current_inv + current_cash
            
            # 2. 定期定額與自動提領機制 (Rebalancing / SIP)
            # 期望目標投資額 = 目前市值 + 今年預計投入
            desired_inv = current_inv + annual_inv
            
            # 向量化動態調整：若總資產 > 0，最多只能投資總資產 (現金不足時自動下修投資部位，等同賣股補貼生活)
            # 若總資產 <= 0 (破產)，投資歸 0，負債由現金部位記帳
            new_inv = np.where(total_w > 0, np.clip(desired_inv, 0, total_w), 0)
            new_cash = total_w - new_inv
            
            inv_wealth[t, :] = new_inv
            cash_wealth[t, :] = new_cash
            wealth[t, :] = total_w
            
        return wealth, inv_wealth

    def run(self):
        cf_rent, cf_buy, prop_val, mort_bal, mort_pmt, pl_pmt, pl_bal = self.build_scenarios()
        
        # 1. 執行流動性模擬，同時抽取出「實際投資部位市值 (inv_amt)」
        mc_rent_invest, mc_rent_inv_amt = self.simulate_wealth_mc(cf_rent, is_investing=True)
        mc_buy_invest, mc_buy_inv_amt = self.simulate_wealth_mc(cf_buy, is_investing=True)
        
        det_rent_no_invest, _ = self.simulate_wealth_mc(cf_rent, is_investing=False)
        det_buy_no_invest, _ = self.simulate_wealth_mc(cf_buy, is_investing=False)
        det_rent_no_invest = det_rent_no_invest[:, 0]
        det_buy_no_invest = det_buy_no_invest[:, 0]
        
        mc_rent_net_worth = mc_rent_invest - pl_bal[:, None]
        mc_buy_net_worth = mc_buy_invest + prop_val[:, None] - mort_bal[:, None] - pl_bal[:, None]
        
        results = pd.DataFrame(index=self.ages)
        results['年齡'] = self.ages
        results['純租_現金流'] = cf_rent
        results['買房_現金流'] = cf_buy
        
        results['純租_無投資'] = det_rent_no_invest - pl_bal
        results['買房_無投資'] = det_buy_no_invest + prop_val - mort_bal - pl_bal
        
        results['純租投資_中位數'] = np.median(mc_rent_net_worth, axis=1)
        results['純租投資_P5'] = np.percentile(mc_rent_net_worth, 5, axis=1)
        results['純租投資_P95'] = np.percentile(mc_rent_net_worth, 95, axis=1)
        
        results['買房投資_中位數'] = np.median(mc_buy_net_worth, axis=1)
        results['買房投資_P5'] = np.percentile(mc_buy_net_worth, 5, axis=1)
        results['買房投資_P95'] = np.percentile(mc_buy_net_worth, 95, axis=1)

        # 隱藏儲存「實際在市場中的投資部位」中位數，專供投資孳息精算使用
        results['純租投資部位_中位數'] = np.median(mc_rent_inv_amt, axis=1)
        results['買房投資部位_中位數'] = np.median(mc_buy_inv_amt, axis=1)
        
        target_age = 65 if 65 in self.ages else self.ages[-1]
        idx_target = np.where(self.ages == target_age)[0][0]
        
        ruin_probs = {
            'buy_inv_65': np.mean(np.any(mc_buy_invest[:idx_target+1, :] < 0, axis=0)) * 100,
            'rent_inv_65': np.mean(np.any(mc_rent_invest[:idx_target+1, :] < 0, axis=0)) * 100,
            'buy_noinv_65': 100.0 if np.any(det_buy_no_invest[:idx_target+1] < 0) else 0.0,
            'rent_noinv_65': 100.0 if np.any(det_rent_no_invest[:idx_target+1] < 0) else 0.0,
            'rent_inv_end': np.mean(np.any(mc_rent_invest < 0, axis=0)) * 100,
            'buy_inv_end': np.mean(np.any(mc_buy_invest < 0, axis=0)) * 100
        }
        
        def calc_median_mdd(net_worth_paths, end_idx):
            paths_subset = net_worth_paths[:end_idx+1, :]
            peaks = np.maximum.accumulate(paths_subset, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdowns = np.where(peaks > 0, (paths_subset - peaks) / peaks, 0)
            drawdowns = np.clip(drawdowns, -1, 0)
            max_drawdowns = np.min(drawdowns, axis=0)
            return abs(np.median(max_drawdowns)) * 100

        mdd_stats = {
            'buy_mdd': calc_median_mdd(mc_buy_net_worth, idx_target),
            'rent_mdd': calc_median_mdd(mc_rent_net_worth, idx_target)
        }

        return results, mort_pmt, ruin_probs, mdd_stats, pl_pmt

# ==========================================
# 2. 頁面設定與 UI 渲染
# ==========================================
st.set_page_config(page_title="量化人生財務模擬器", page_icon="📈", layout="wide")
st.title("📈 量化人生財務模擬器 (Institutional Monte Carlo Edition)")
st.markdown("融合**精算科學 (Actuarial Science)**與**隨機微積分 (Stochastic Calculus)**，真實呈現市場動態中的量化邊界。")

with st.sidebar:
    st.header("⚙️ 量化參數設定")
    
    st.subheader("👤 基本財務")
    p_age = st.number_input("目前年齡", min_value=20, max_value=60, value=30, step=1)
    p_retire = st.number_input("預計退休年齡", min_value=40, max_value=80, value=65, step=1)
    
    p_capital = st.number_input("起始流動資金 (萬元)", min_value=0, value=100, step=10, help="目前的現金存款")
    p_exist_invest = st.number_input("目前已投資部位 (萬元)", min_value=0, value=100, step=10, help="目前已經投入股市/基金等市場的資產")
    
    p_inflation = st.number_input("年通膨/薪資成長率 (%)", min_value=0.0, value=2.0, step=0.5) / 100.0
    
    p_salary = st.number_input("目前平均月薪 (萬元)", min_value=0.0, value=8.0, step=0.5, help="實領金額")
    p_expense = st.number_input("目前月開銷 (萬元)", min_value=0.0, value=3.0, step=0.1)
    p_rent = st.number_input("目前月房租 (萬元)", min_value=0.0, value=2.0, step=0.1)

    st.markdown("---")
    st.subheader("🛡️ 退休金與勞退帳戶 (台灣勞退機制)")
    p_pension_current = st.number_input("目前已提撥勞退本金 (萬元)", min_value=0.0, value=30.0, step=10.0, help="勞退個人專戶目前累積之本金與收益")
    p_pension_monthly = st.number_input("每月持續提撥額 (萬元)", min_value=0.0, value=0.6, step=0.1, help="含雇主6%與自提。此資金獨立累積，不扣除上方月薪之流動現金。")
    p_pension_return = st.number_input("勞退基金保證年化報酬率 (%)", value=2.0, step=0.1) / 100.0
    
    st.markdown("---")
    st.subheader("💳 現有銀行貸款 (信貸/車貸)")
    p_pl_balance = st.number_input("目前剩餘貸款本金 (萬元)", min_value=0.0, value=0.0, step=10.0, help="例如：個人信貸、車輛貸款等剛性債務")
    p_pl_years = st.number_input("剩餘還款年限 (年)", min_value=0, max_value=20, value=0, step=1)
    p_pl_rate = st.number_input("貸款年利率 (%)", min_value=0.0, value=3.0, step=0.1) / 100.0
    
    st.markdown("---")
    st.subheader("📊 投資市場動態")
    p_monthly_inv = st.number_input("每月投資金額 (萬元)", min_value=0.0, value=3.0, step=0.1, help="每月從現金流中扣除並投入市場的金額 (即定期定額)。若年度現金流不足以負擔此金額，系統將自動調整；若生活費不足，將自動變賣投資部位補貼。")
    p_return = st.number_input("預期年化報酬率 (μ) (%)", value=7.0, step=0.5, help="長線投資組合的預期年化報酬率。例如：全球股票型 ETF 約 7%~9%。") / 100.0
    p_volatility = st.number_input("年化波動率 (σ) (%)", value=15.0, step=0.5, help="反映市場震盪的風險程度。例如 S&P 500 長期歷史平均年化波動率約為 15%~18%，台股大盤約 16%~20%，全市場投資級債券約 5%~7%。") / 100.0
    p_paths = st.selectbox("蒙地卡羅路徑數", options=[100, 500, 1000, 5000, 10000], index=2, help="模擬未來可能發生的平行宇宙數量。路徑數越高，極端風險 (P5) 與破產率的統計評估越精準，但運算時間較長。標準量化回測建議至少 1000 條以上。")
    
    st.markdown("---")
    st.subheader("🏠 購屋與貸款模組")
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
    '起始年齡': p_age, '退休年齡': p_retire, '起始資金': p_capital, '現有投資': p_exist_invest, '通膨率': p_inflation,
    '月薪': p_salary, '月開銷': p_expense, '月房租': p_rent,
    '勞退目前提撥': p_pension_current, '勞退每月提撥': p_pension_monthly, '勞退報酬率': p_pension_return,
    '信貸餘額': p_pl_balance, '信貸年限': p_pl_years, '信貸利率': p_pl_rate,
    '每月投資': p_monthly_inv, '預期報酬': p_return, '預期波動率': p_volatility, '模擬路徑數': p_paths,
    '買房年齡': p_buy_age, '房價': p_house_price, '頭期款': p_down_pmt,
    '房產年增值': p_house_appr, '房貸年限': p_loan_years, '寬限期': p_grace_years, '房貸利率': p_loan_rate
}

model = LifeFinancialALM(params)
df_res, mort_pmt, ruin_probs, mdd_stats, pl_pmt = model.run()

# ==========================================
# 3. 儀表板與數據視覺化 (不改變原架構)
# ==========================================
target_age = 65 if 65 in df_res.index else df_res.index[-1]
target_data = df_res.loc[target_age]

# ----------------- 計算 65 歲預期投資孳息 -----------------
# 投資組合年孳息 = 實際在市場中的投資部位市值 * 預期年化報酬率(μ)
rent_inv_return = target_data['純租投資部位_中位數'] * p_return
buy_inv_return = target_data['買房投資部位_中位數'] * p_return

st.subheader(f"📊 {target_age}歲 財務健康度總覽 (淨資產與流動性風險)")

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
st.markdown("實線為**純儲蓄（無投資）**之基準；虛線為市場動態下之**中位數**，陰影區間代表 **90% 信心水準**。")

fig_nw = go.Figure()

# 買房情境
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_P95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_P5'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(52, 199, 89, 0.2)', name='買房+投資 (90% 區間)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房投資_中位數'], mode='lines', line=dict(color='#34C759', width=3, dash='dash'), name='買房+投資 (中位數)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['買房_無投資'], mode='lines', line=dict(color='#FF3B30', width=3), name='買房純儲蓄'))

# 純租情境
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_P95'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_P5'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 122, 255, 0.2)', name='純租+投資 (90% 區間)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租投資_中位數'], mode='lines', line=dict(color='#007AFF', width=3, dash='dash'), name='純租+投資 (中位數)'))
fig_nw.add_trace(go.Scatter(x=df_res['年齡'], y=df_res['純租_無投資'], mode='lines', line=dict(color='#8E8E93', width=3), name='純租純儲蓄'))

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
fig_cf.add_hline(y=0, line_dash="dash", line_color="#8E8E93", annotation_text="收支平衡線", annotation_position="bottom right")
fig_cf.update_layout(
    plot_bgcolor='white', paper_bgcolor='white', hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年齡'),
    yaxis=dict(showgrid=True, gridcolor='#E5E5EA', title='年度淨現金流 (萬元)')
)
st.plotly_chart(fig_cf, use_container_width=True)

# ----------------- 財務洞察 -----------------
with st.expander("📝 量化專家診斷報告 (點擊展開)", expanded=True):
    st.write(f"""
    ### 深度量化診斷結果
    
    | 診斷維度 | 量化回測事實 | 財務意義 |
    | :--- | :--- | :--- |
    | **破產風險評估 <br> (Probability of Ruin)** | 終老純租破產率：**{ruin_probs['rent_inv_end']:.1f}%** <br> 終老買房破產率：**{ruin_probs['buy_inv_end']:.1f}%** | 嚴格依循路徑依賴演算法，生命週期中「流動性跌破 0」即觸發實質違約。買房情境在寬限期結束後的現金流斷層，是誘發流動性危機的最大震央。 |
    | **最大回撤與波動 <br> (Max Drawdown)** | 買房投資 MDD：**-{mdd_stats['buy_mdd']:.1f}%** <br> 純租投資 MDD：**-{mdd_stats['rent_mdd']:.1f}%** | 純租情境的淨資產完全暴露於市場波動風險；而買房情境藉由低波動實體資產（年增值 {p_house_appr*100:.1f}%），實質上提供了投資組合下檔保護。 |
    | **長壽風險與勞退 <br> (Longevity & Pension)** | 65-84歲勞退(年)：**{model.payout_annual:.1f} 萬** <br> 65歲後預期投資孳息(年)：**純租 {rent_inv_return:.1f} 萬 / 買房 {buy_inv_return:.1f} 萬** | 依台灣勞退新制規定，月領年金精算至 84 歲為止。圖表 2 中可明顯觀察到 **85 歲的二次現金流斷崖**，此後將大幅考驗自身投資組合的抗摔能力。 |
    | **房產槓桿與流動性 <br> (Real Estate Leverage)** | 房貸年付 (攤還期)：**{np.max(mort_pmt):.1f} 萬** <br> 房貸總年限：**{p_loan_years} 年** | 房產雖具備抗通膨與強制儲蓄特性，但本息攤還將大幅排擠定期的「每月投資金額」。若遇市場長期空頭且逢寬限期結束，房貸的剛性支出將嚴重侵蝕剩餘現金流。 |
    | **消費負債拖累 <br> (Consumer Debt Drag)** | 銀行貸款年付：**{pl_pmt[0]:.1f} 萬** <br> 貸款剩餘年限：**{p_pl_years} 年** | 剛性債務（信貸/車貸）將在初期產生「現金流拖累」。由於淨資產計算中已嚴格扣除此負債，若其貸款利率大於投資組合底層期望值，將顯著放大早夭期的流動性違約風險。 |
    """)