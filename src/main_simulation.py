import numpy as np
import random
import matplotlib.pyplot as plt
from VolModel.EWMAModel import EWMAModel
from VolModel.GARCHModel import GARCHModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution
import utils
import scipy as scp
import numpy as np

def simulate_returns(n, lmda, dist):
    # initiate the DGP
    returns = [0.01]
    variance = [np.power(0.01, 2)]
    # EWMA + Normal distribution + random shocks
    for i in range(1, n):
        var_t = (1 - lmda) * np.power(returns[i-1], 2) + lmda * variance[i-1]
        variance.append(var_t)
        returns.append(np.sqrt(var_t) * dist.random_draw())
        
    return np.array(returns), np.array(variance)

# DGP
true_dist = StudentsDistribution(df=3)
true_model = EWMAModel(true_dist)
true_param = 0.98

# mispecifed case
# pred_dist = NormalDistribution()
# pred_model = GARCHModel(pred_dist)

# correctly specified case
pred_dist = StudentsDistribution()
pred_model = EWMAModel(pred_dist)

n_iter = 0
VaR_level = 0.05

results = {
    'MLE' : [],
    'FV' : []
    }

for i in range(0, n_iter):
    print(i)
    
    true_returns, true_variance = simulate_returns(5000, true_param, true_dist)
    
    ml_param = pred_model.fit_mle(true_returns, show=False)
    fv_param = pred_model.fit_fv(true_returns, VaR_level, show=False)
    
    ml_VaR = pred_model.get_valueatrisk(true_returns, ml_param, VaR_level)
    fv_VaR = pred_model.get_valueatrisk(true_returns, fv_param, VaR_level)
    
    ml_pval, ml_lrcc = utils.test_independence(true_returns, ml_VaR, VaR_level)
    fv_pval, fv_lrcc = utils.test_independence(true_returns, fv_VaR, VaR_level)
    
    if ml_lrcc < 1000 and fv_lrcc < 1000:
        results['MLE'].append(ml_lrcc)
        results['FV'].append(fv_lrcc)
    
plt.figure(figsize=(12, 7))
plt.hist(results['MLE'], label='MLE', facecolor='black', bins=25, alpha=1.0)
plt.hist(results['FV'], label='FV', facecolor='grey', bins=25, alpha=1.0)
plt.legend(fontsize=20)
plt.ylabel('count', fontsize=15)
plt.xlabel('LRcc', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(r'D:\PhD\Var\Results\\LRcc_simul.eps', format='eps', dpi=300)
plt.show()
    
print(f"LRCC average (MLE) : {np.mean(results['MLE']):.2f}")
print(f"LRCC average (FV) : {np.mean(results['FV']):.2f}")

r = 0.95
ml_reject = sum((np.array(results['MLE']) > scp.stats.chi2.ppf(q=r, df=2)).astype('int')) / n_iter
print(f"rejection of H0 (MLE) at {1-r:.2f} : {ml_reject * 100:.2f} %")
fv_reject = sum((np.array(results['FV']) > scp.stats.chi2.ppf(q=r, df=2)).astype('int')) / n_iter
print(f"rejection of H0 (FV) at {1-r:.2f} : {fv_reject * 100:.2f} %")

# =============================================================================
# 
# =============================================================================

inpPath = 'D:\\PhD\\VaR\\Data'
retTbl = utils.prepare_data(inpPath, returns='pct')
returns = np.array(retTbl[['SPX Index']]).reshape(-1,)

lambda_array = np.linspace(0.90, 1.0, 5000)
VaR_level = 0.05
results = []

dist = NormalDistribution()
model = EWMAModel(dist)

for lmda in lambda_array:
    VaR = model.get_valueatrisk(returns, [lmda], VaR_level)
    violations = utils.count_violations(returns, VaR)
    b_loglik = utils.binomial_loglik(violations, VaR_level)
    m_loglik = utils.markov_loglik(violations)
    results.append([b_loglik, m_loglik])

results = np.array(results)
    
plt.figure()
plt.plot(lambda_array, results[:, 0], label='LR_uc', c='dimgrey', ls='-.')
plt.plot(lambda_array, results[:, 1], label='LR_ind', c='darkgrey', ls='-')
plt.plot(lambda_array, results[:, 0] + results[:, 1], label='LR_cc', c='black', ls='--')
plt.legend(loc='upper left')
plt.xlabel('lambda')
plt.ylabel('Likelihood Ratio')
plt.savefig('D:\\PhD\\VaR\\Results\\gridsearchlambda.eps', format='eps', dpi=300)
plt.plot()
