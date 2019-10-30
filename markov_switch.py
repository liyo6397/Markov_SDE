import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans


class Parameter_measure:
    def __init__(self, returns, volume):
        self.xs = returns
        self.volume = volume
        self.num_state = 2
        self.tol = 0.01
        self.max_iter = 100
        self.pis, self.mus, self.sigmas, self.transit_prob = self.ini_sde()

    def ini_sde(self):
        n, p = self.xs.shape
        test_window = len(self.xs)
        volume_group = np.zeros(self.num_state)
        pis = np.zeros(self.num_state)
        mus = np.zeros((self.num_state, p))
        sigmas = np.array([np.eye(2)] * 2)

        observed = np.array(self.volume)
        kmeans = KMeans(n_clusters=self.num_state, random_state=0).fit(observed)
        idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(self.num_state)



        for k in range(self.num_state):
            ix = [x for x in range(test_window) if lut[kmeans.labels_][x] == k]
            xs_group = self.xs[ix]
            pis[k] = len(xs_group) / len(self.xs)
            mus[k] = np.mean(xs_group)
            volume_group[k] = np.mean(self.volume[ix])


        transit_prob = np.array([[0.6, 0.4], [0.5, 0.5]])

        return pis, mus, sigmas, transit_prob

    def em_gmm_orig(self):

        n, p = self.xs.shape
        k = 2

        ll_old = 0
        for i in range(self.max_iter):

            # E-step
            ws = np.zeros((k, n))
            for j in range(len(mus)):
                for i in range(n):
                    ws[j, i] = pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
            ws /= ws.sum(0)

            # M-step
            pis = np.zeros(k)
            for j in range(len(mus)):
                for i in range(n):
                    pis[j] += ws[j, i]
            pis /= n

            mus = np.zeros((k, p))
            for j in range(k):
                for i in range(n):
                    mus[j] += ws[j, i] * self.xs[i]
                mus[j] /= ws[j, :].sum()

            sigmas = np.zeros((k, p, p))
            for j in range(k):
                for i in range(n):
                    ys = np.reshape(self.xs[i] - mus[j], (2, 1))
                    sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
                sigmas[j] /= ws[j, :].sum()

            # update complete log likelihoood
            ll_new = 0.0
            for i in range(n):
                s = 0
                for j in range(k):
                    s += pis[j] * mvn(mus[j], sigmas[j]).pdf(self.xs[i])
                ll_new += np.log(s)

            if np.abs(ll_new - ll_old) < self.tol:
                break
            ll_old = ll_new

        return ll_new, pis, mus, sigmas


# Read data
def main():
    stock_quote = input('Enter a stock quote from NASDAQ (e.j: AAPL, FB, GOOG): ').upper()
    FILE_NAME = 'data/{}.csv'.format(stock_quote)
    df_raw = pd.read_csv(FILE_NAME, parse_dates=True, index_col=0)
    df = df_raw.sort_index()
    x = np.array(df.index)
    date = pd.to_datetime(x)
    date_num = mdates.date2num(date)

    close_v = np.array(df['Close'])
    df['fracChange'] = (df['Close'] - df['Open']) / df['Open']
    returns = np.array(df['fracChange'])
    returns = close_v.reshape(-1,1)
    volume = np.array(df['Volume'])
    volume = volume.reshape(-1, 1)

    parameters = Parameter_measure(returns, volume)
    ll_new, pis, mus, sigmas = parameters.em_gmm_orig


if __name__ == "main":
    main()
