from VolModel.GARCHModel import GARCHModel
from VolModel.EWMAModel import EWMAModel
from VolModel.GJRGARCHModel import GJRGARCHModel
from Distribution.NormalDistribution import NormalDistribution
from Distribution.StudentsDistribution import StudentsDistribution
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import os

def boxplot_test(test_array, y_name, config_name):
    output_dir = f'C:\\Users\\Lucas\\Desktop\\PhD\\VaR\\Results\\{config_name}\\'
    os.makedirs(output_dir, exist_ok=True)
    title = f'{y_name} {config_name}'
    file_name = f'{title}.eps'
    
    plt.figure()
    for j in [0, 1]:
        plt.boxplot(test_array, tick_labels=['MLE', 'FV'])
        plt.title(title)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=10)
        plt.savefig(output_dir + file_name, format='eps', dpi=300)
    plt.show()
    
def backtest(test_array, conf_level):
    results = []
    for j in [0, 1]:
        pass_test = float(round(sum(np.array(test_array[j]) < conf_level) / len(test_array[j]), 2))
        results.append(pass_test)
        
    return results
    
# =============================================================================
# 
# =============================================================================

inpPath = 'C:\Users\Lucas\Desktop\PhD\VaR\Data'
retTbl = utils.prepare_data(inpPath, returns='pct')
asset_names = retTbl.columns

VaR_level = 0.025
dist = NormalDistribution()
model = GJRGARCHModel(dist)
config = f'{model.config_name()} {str(VaR_level)}'

kupiec, indep, duration, dq = [[], []], [[], []], [[], []], [[], []]
extime_mle, extime_fv = [], []

for asset in asset_names:
    print(asset)
    returns = np.array(retTbl[asset])
    train, test = utils.train_test_split(returns, 0.8)

    start_time = time.time()
    model.fit_mle(train, show=False)
    extime_mle.append(time.time() - start_time)
    params_mle = model.mle_params
    VaR_mle = model.get_valueatrisk(test, params_mle, VaR_level)
    
    start_time = time.time()
    model.fit_fv(train, VaR_level, show=False)
    extime_fv.append(time.time() - start_time)
    params_fv = model.fv_params
    VaR_fv = model.get_valueatrisk(test, params_fv, VaR_level)
    
    VaRs = [VaR_mle, VaR_fv]
    
    for j in [0, 1]:
        kupiec[j].append(utils.test_kupiec(test, VaRs[j], VaR_level))
        indep[j].append(utils.test_independence(test, VaRs[j], VaR_level)[0])
        duration[j].append(utils.test_duration(test, VaRs[j]))
        dq[j].append(utils.test_dq(test, VaRs[j], 4, 1, VaR_level))

boxplot_test(kupiec, 'Unconditional Coverage Test', config)
boxplot_test(indep, 'Conditional Coverage Test', config)
boxplot_test(duration, 'Duration Test', config)
boxplot_test(dq, 'Dynamic Quantiles Test', config)
boxplot_test([extime_mle, extime_fv], 'Execution Time (seconds)', config)

print('Kupiec')
print('0.05', backtest(kupiec, 0.05))
print('0.01', backtest(kupiec, 0.01))
print('Christoffersen')
print('0.05', backtest(indep, 0.05))
print('0.01', backtest(indep, 0.01))
print('Duration')
print('0.05', backtest(duration, 0.05))
print('0.01', backtest(duration, 0.01))
print('DQ')
print('0.05', backtest(dq, 0.05))
print('0.01', backtest(dq, 0.01))
print('Ex Time')
print(f'{np.mean(extime_mle):.3f}', f'{np.mean(extime_fv):.3f}')


