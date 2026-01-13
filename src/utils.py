import numpy as np
import pandas as pd
import scipy as scp

def first_difference(df, how):
    # computes returns using either percentage change or logarirthmic returns
    col_names = list(df.columns)
    idx = list(df.index)[1:]

    if how == 'pct':
        new_df = df.pct_change(1) * 100
        new_df.drop(new_df.index[0], axis=0, inplace=True)
    elif how == 'log':
        new_df = pd.DataFrame(np.diff(np.log(df), axis=0) * 100, columns=col_names)

    new_df.columns = col_names
    new_df.index = idx

    return new_df


def make_portfolio(data, asset_type):
    temp_array = [0] * len(data)
    n_asset = 0

    for col in data.columns:
        if asset_type in col:
            n_asset += 1
            temp_array = temp_array + data[col]

    return np.array(temp_array) / n_asset


def prepare_data(path, returns='pct'):
    # reads the data from an Excel file and imports it as a pandas DataFrame
    inpTbl = pd.read_excel(path + r'\inpAllFle_updated.xlsx',
                           header=0,
                           index_col=0,
                           na_values='null',
                           parse_dates=True)
    inpTbl.drop(['CL1 Comdty'], axis=1, inplace=True)
    retTbl = first_difference(inpTbl, how='pct')

    # remove first row
    retTbl = retTbl.iloc[1:]

    # remove any asset that have at least 1 year of NAs
    to_remove = []
    start_date = 260
    for col in retTbl.columns:
        if pd.isnull(retTbl[col].iloc[start_date]):
            to_remove.append(col)
    retTbl.drop(to_remove, axis=1, inplace=True)

    # remove remaining NAs
    retTbl.dropna(inplace=True)

    # create the portofolios
    for asset_type in ['Index', 'Curncy', 'Comdty']:
        new_col_name = asset_type + '_ptfl'
        retTbl[new_col_name] = make_portfolio(retTbl, asset_type)

    return retTbl

def weibull_pdf(a, b, D):
    # computes the weibull pdf
    val = np.array(a ** b * b * D ** (b - 1) * np.exp(-(a * D) ** b))
    val[val == 0] = 1e-6

    return val

def weibull_cdf(a, b, D):
    # computes weibull cdf
    val = 1 - np.exp(-(a * D) ** b)

    if val != 0 and val != 1:
        return val
    elif val == 1:
        return 1 - 1e-6
    else:
        return 1e-6

def test_duration(returns, VaR):
    violations = count_violations(returns, VaR)
    
    def duration(violations):
        # computes the duration between each violation
        # also computes the "censored" factors for the first and last observations
        # see Christoffersen & Pelletier 2004

        count = 1
        D = []
        C = []

        for i in range(0, violations.shape[0]):
            if i == 0 or i == violations.shape[0] - 1:
                if violations[i] == 0:
                    C.append(1)
                else:
                    C.append(0)

            if i == violations.shape[0] - 1 and violations[i] == 0:
                D.append(count)

            if violations[i] == 0:
                count += 1
            else:
                D.append(count)
                count = 1

        return np.array(D), np.array(C)
    
    D, C = duration(violations)

    def duration_loglik(params, *args):
        # computes the log likelihood of observing the durations sequence
        # see Christiffersen & Pelletier 2004

        # returns the negative log likelihood
        b = params[0]
        D, C = args
        C_1 = C[0]
        C_NT = C[1]
        N = D.shape[0]
        a = np.power((N - C_1 - C_NT) / sum(np.power(D, b)), (1 / b))

        loglik = (C_1 * np.log(1 - weibull_cdf(a, b, D[0]))
                  + (1 - C_1) * np.log(weibull_pdf(a, b, D[0]))
                  + sum(np.log(weibull_pdf(a, b, D[1: -1])))
                  + C_NT * np.log(1 - weibull_cdf(a, b, D[-1]))
                  + (1 - C_NT) * np.log(weibull_pdf(a, b, D[-1])))

        return -loglik

    opt = scp.optimize.minimize(duration_loglik,
                                x0=[1.0],
                                args=(D, C),
                                bounds=[(0.001, 10.0)],
                                method='SLSQP')

    b = opt.x[0]

    h0 = -duration_loglik([1], D, C)
    h1 = -duration_loglik([b], D, C)
    ratio = 2 * (h1 - h0)

    return 1 - chi2_cdf(X=ratio, k=1)

def test_independence(returns, VaR, VaR_level):
    # computes the p value for idependence of hits
    # see Christoffersen 1998
    # uses the likelihood ratio from markov_loglik
    violations = count_violations(returns, VaR)
    b_loglik = binomial_loglik(violations, VaR_level)
    m_loglik = markov_loglik(violations)
    ll_ratio = b_loglik + m_loglik
    p_val = 1 - chi2_cdf(X=ll_ratio, k=2)
    
    return p_val, ll_ratio

def count_violations(returns, VaR):
    return (returns < VaR).astype('int')

def chi2_pdf(X, k):
    # computes the chi2 pdf
    gamma = scp.special.gamma(k / 2)

    return (1 / (2 ** (k / 2) * gamma)) * X ** (k / 2 - 1) * np.exp(-X / 2)


def chi2_cdf(X, k):
    return scp.special.gammainc(k / 2, X / 2)


def test_kupiec(returns, VaR, VaR_level):
    violations = count_violations(returns, VaR)
    ll_ratio = binomial_loglik(violations, VaR_level)

    return 1 - chi2_cdf(ll_ratio, k=1)


def test_dq(returns, VaR, v_lag, f_lag, VaR_level):
    # Engle & Manganelli test
    violations = count_violations(returns, VaR)
    p, q, n = v_lag, f_lag, violations.shape[0]
    pq = max(p, q - 1)
    y = violations[pq:] - VaR_level
    x = np.zeros((n - pq, 1 + p + q))
    x[:, 0] = 1
    
    for i in range(p):  # Lagged hits
        x[:, 1 + i] = violations[pq - (i + 1):-(i + 1)]

    for j in range(q):  # Actual + lagged VaR forecast
        if j > 0:
            x[:, 1 + p + j] = VaR[pq - j:-j]
        else:
            x[:, 1 + p + j] = VaR[pq:]

    try:
        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) /\
            (VaR_level * (1 - VaR_level))
        p_dq = 1 - scp.stats.chi2.cdf(lr_dq, 1 + p + q)
    
    except:
        p_dq=1e-6
    return p_dq


def markov_loglik(data):
    
    prev_state = data[:-1]
    next_state = data[1:]
    
    pairs = 2 * prev_state + next_state
    pairs = pairs.astype('int')
    
    n_00, n_01, n_10, n_11 = np.bincount(pairs, minlength=4)
    
    # additive smoothing for numerical stability
    n_00 += 1e-6
    n_10 += 1e-6
    n_01 += 1e-6
    n_11 += 1e-6
    
    # transition probabilities
    c_00 = n_00 / (n_00 + n_01)
    c_01 = n_01 / (n_00 + n_01)
    c_10 = n_10 / (n_10 + n_11)
    c_11 = n_11 / (n_10 + n_11)
    
    pi_2 = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)
    
    # loglikelihood ratio
    h1 = n_00 * np.log(c_00) + n_01 * np.log(c_01) + n_10 * np.log(c_10) + n_11 * np.log(c_11)
    h0 = (n_00 + n_10) * np.log(1 - pi_2) + (n_01 + n_11) * np.log(pi_2)

    return -2 * (h0 - h1)


def binomial_loglik(data, p):

    # computes the likelihood ratio of observing 'k' violations
    # assuming the data comes from a B(n, VaR_level) distribution
    # versus the likelihood that the data comes from a B(n , k/n) distribution
    # formula for the likelihood of a Binomial distribution can be found here:
    # https://en.wikipedia.org/wiki/Binomial_distribution

    # returns the log-likelihood ratio lL(B(n, VaR_level)) / lL(B(n, k/n))
    n = data.shape[0]
    k = max(int(sum(data)), 1e-6)
    # we use the log-likelihood to avoid numerical under/over flow
    # and other floating points precision related issues
    h0 = k * np.log(p) + (n - k) * np.log(1 - p)
    h1 = k * np.log(k / n) + (n - k) * np.log(1 - (k / n))

    return -2 * (h0 - h1)

def train_test_split(returns, train_size):
    # splits the return series into a training and testing set
    split_idx = int(train_size * returns.shape[0])
    train = returns[: split_idx]
    test = returns[split_idx:]

    return train, test