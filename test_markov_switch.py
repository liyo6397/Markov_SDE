import unittest
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from scipy.stats import multivariate_normal as mvn

from Markov_SDE.markov_switch import Parameter_measure


class Test_MarkovSwitch(unittest.TestCase):

    def setUp(self):
        df_raw = pd.read_csv('data/FB.csv', parse_dates=True, index_col=0)
        self.df = df_raw.sort_index()
        self.close_v = np.array(self.df['Close'])
        self.df['fracChange'] = (self.df['Close'] - self.df['Open']) / self.df['Open']
        self.returns = np.array(self.df['fracChange'][:10]).reshape(-1,1)
        self.volume = np.array(self.df['Volume'][:10]).reshape(-1,1)
        self.date = pd.to_datetime(self.close_v)
        self.xs = self.close_v.reshape(-1,1)
        self.date_num = mdates.date2num(self.date)
        self.transit_prob = np.array([[0.6, 0.4], [0.5, 0.5]])
        self.pars = Parameter_measure(self.returns, self.volume)

    def test_ini_sde(self):
        pis, mus, sigmas,trans_prob = self.pars.ini_sde()


        print("pi",pis)
        print("mus",mus)
        print("sigmas",sigmas)
        print("transit",trans_prob)



    def test_em_gmm_orig(self):
        ll_new, pis, mus, sigmas = self.pars.em_gmm_orig()

        print(ll_new)
        print(pis)
        print(mus)
        print(sigmas)
