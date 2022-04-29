"""
Module that implements a framework to select suitable pairs of assets for statistical arbitrage.
Based on the paper 'Enhancing a Pairs Trading strategy with the application of Machine Learning'(Saramento et al.,2020)
source: http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf
"""

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import itertools as it

import plotly.express as px

from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE
  
class PairsClustering:    
    """
    Class to use principal component analysis to reduce dimensionality and apply the optics algorithm to
    find clusters of the data.

    """
    
    def plot_pca_explained_variation(self,df,n_components =20):
        """
        Fuction that plots the cumululative variation explained of the data per principal component.  

        :param df : (pandas.Dataframe) Dataframe of the data.
        :param n_componets: (int) The number of principal components to include. By defualt it is 20.
                                  

        """  
        # fit the data to the PCA
        pca = PCA(n_components).fit(df)

        #sum the cumulutaive varianve explained
        variation_explained_cum  = np.cumsum(pca.explained_variance_ratio_)* 100

        #plot the variation explained
        fig = px.bar(x=range(1, n_components + 1),
        y=variation_explained_cum,
        labels={"x": "# Components", "y": " % Explained Variance"} , 
            title = 'Variation explained by # of PC')
        fig.show()
        
    def principal_components_vectors(self,df,number_of_principal_components):
        """ 
        Function that returns the principal component vectors.

        :param df : (pandas.Dataframe) Dataframe of the data.
        :param n_componets: (int) : The number of principal components to include.
        :return (numpy.ndarray) : The prinicpal component vectors.
        """
        # initiliace the PCA and fit it
        pca = PCA(n_components = number_of_principal_components)
        pca.fit(df)

        # standardize the principal components                             
        principal_components = preprocessing.StandardScaler().fit_transform(pca.components_.T)

        return principal_components



    def cluster_OPTICS(self,pca_components, df_returns, min_samples = 3, max_eps=2, ):
        """
        Function to find clustesr using the OPTICS algorithm

        :param pca_components  : (numpy.darray) Prinicipal component vectors
        :param df_returns : (pandas.DataFrame)  DataFrame of returns where the columns 
                                                are the tickers and the rows are the daily prices.
        :param min_samples : (int) The minimum number of samples for a point to be considered a core point 
                                   (the minimum cluster size).
        :return clustered_numbers : (pandas.DataFrame) Dataframe of the tickers and to which clusters they belong. 
        :return clustered_numbers_all : (numpy.darray) Index value of the clusters.
        :return grouped_clusters : (pandas.DataFrame) Dataframe of the clusters and which stocks each cluster contains
        """

        # initialize OPTICS and fit it 
        clustering = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=0.05, metric='euclidean', cluster_method = 'xi')
        clustering .fit(pca_components)

        # gettting the cluster labels to which each stock belongs
        clustered_numbers_all = clustering.labels_

        # contructing dataframe for the cluster that the stocks belong to 
        tickers = pd.DataFrame(df_returns.columns)
        cluster =  pd.DataFrame(clustered_numbers_all.flatten())
        clustered_numbers = pd.concat([tickers,cluster],axis =1)
        clustered_numbers.columns = ['Tickers','Cluster']                            
        clustered_numbers = pd.DataFrame.dropna(clustered_numbers[clustered_numbers_all != -1])

        # constructing dataframe group of clusters
        grouped_df = clustered_numbers.groupby('Cluster')
        grouped_lists = grouped_df['Tickers'].apply(list)
        grouped_clusters = grouped_lists.reset_index() 
        grouped_clusters['Count'] = grouped_clusters['Tickers'].apply(len)
        grouped_clusters['Cluster'] = grouped_clusters['Cluster'].apply(lambda x :  x+1)
        grouped_clusters.index += 1

        # calculating the number of pairs to evaluate
        pairs_numbers = np.sum(grouped_clusters['Count'].apply(lambda x :math.comb(x,2)))
        print("Number of pairs to evalute is {}".format(pairs_numbers))

        # calculating the number of clusters discovered
        num_clusters  = grouped_clusters.shape[0]
        print("Clusters discovered: {}".format(num_clusters))

        return  clustered_numbers ,clustered_numbers_all, grouped_clusters 


    def plot_TSNE(self,principal_components,  clustered_numbers , clustered_numbers_all , learning_rate = 500,  perplexity=30):
        """
        Function to plot the TSNE graph to visualize clusters in 2 dimensions.

        :param principal_components : (numpy.darray) Principal component vectors.
        :param clustered_number : (pandas.DataFrame) Dataframe of the tickers to which clusters they belong. 
        :param clustered_number_all : (numpy.darray) Index value of the clusters.
        :parmam learning_rate: (int) Determines the step size at each iteration while moving toward a minimum of a loss function.
        :param perplexity : (int) Paramter that balances attention between local and global aspects the data. 
        """

        # initialize TSNE and fit to data
        pc_TSNE = TSNE(learning_rate=learning_rate, perplexity=perplexity,
                       random_state=500).fit_transform(principal_components)

        # construsting dataframe of the x-axis and y-axis values for the TSNE of tickers in the clusters
        df = pd.DataFrame(pc_TSNE[(clustered_numbers_all!=-1), 0:2])

        # getting the index value of only the tickers that are in clusters    
        clusters =  clustered_numbers_all[(clustered_numbers_all != -1)]

        # appending dataframe df to include the tickers(labels) and the clusters (colours)
        df['Tickers'] = list(clustered_numbers['Tickers'])
        df['Clusters'] = clusters
        df = df.sort_values('Clusters')
        df['Clusters'] = df['Clusters'].astype(str)
        size = 300
        df['Size']=  [size]*len(clusters)

        #plotting the TSNE graph
        fig = px.scatter(df, x=0,y=1, color = 'Clusters' ,
             text = 'Tickers', size = 'Size' ,
             title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)').update_layout(xaxis_title="1-D",
                                                                                          yaxis_title="2-D")

        fig.show()
        pass


class PairSelectionCriteria:
    """
    Class to validate whether certain conditions are met in order for pairs of stocks to be suitable for pairs trading.

    """
    
    def coint_stats(self,X,Y):

        """ 
        Function to calculate whether two time series are cointegrated as well as the spread of the log time series.
        The function calculates for both pairs X and Y as the independant variable in order to reduce test reliance on
        the depedant variable.

        :param X,Y : (pandas.Series) Series of prices that need to be checked for cointegration.
        :return coint_bool : (boolean) Returns whethers or not the two time series are cointegrated.
        :return spread : (numpy.darra) The spread between the two series.
        :return dependant: (str) Returns which series is the dependant variable.

        """
        #caculate the p-value of the Augmented Dickey Fuller test
        x_stationary_pvalue = round(adfuller(X)[1],3)
        y_stationary_pvalue = round(adfuller(Y)[1],3)

        #initialize variables
        coint_bool = False
        dependant = 'X'
        spread = [0]*len(X)

        #add constant to series in order to fit
        X_cont = sm.add_constant(X)    
        Y_cont = sm.add_constant(Y)

        # use OLS to fit the model and estimate parameter
        model_X = sm.OLS(Y, X_cont).fit()
        model_Y = sm.OLS(X, Y_cont).fit()

        #calculate the spreads between the models
        spread_X = Y - model_X.params[1] * X - model_X.params[0] 
        spread_Y = X - model_Y.params[1] * Y - model_Y.params[0]

        #check that both series are non-stationary at a 5% percent significance level
        if (x_stationary_pvalue >= 0.05) & (y_stationary_pvalue >= 0.05):

            #caculate the p-value of the Augmented Dickey Fuller test for the spreads    
            spread_x_stationary_pvalue = round(adfuller(spread_X)[1],3)
            spread_y_stationary_pvalue = round(adfuller(spread_Y)[1],3)

            # check with pairs of depenant and independant values have the lowest t-statistc(p-value)    
            if spread_x_stationary_pvalue <= spread_y_stationary_pvalue:
                spread_pvalue = spread_x_stationary_pvalue
                spread = spread_X 
                dependant = 'X'

            else:
                spread_pvalue = spread_x_stationary_pvalue
                spread = spread_Y
                dependant = 'Y'
                
            # check whether the spread is stationary  - then the two series are conintegrated
            if  spread_pvalue <= 0.05:
                coint_bool = True

        return coint_bool, spread , dependant

    def hurst_exponent(self,series,max_lag = 100):
        """
        Function to calculate to hurst exponent of series.
        Simplest method based on the diffusive behaviour of the variance of prices.

        :param : (pandas.Series) Series of prices .
        :param max_lag : (int) Arbitrary value for the maximum lag.
        :return : (float) The hurst exponent.
        """

        # calculate the range of lags
        lags = range(1,max_lag+1)
        
        #creat a lagged series
        series_lag = np.roll(series, max_lag)
       
        #calculate the variances of the lagged series differences
        var_tau = []
        length_s = len(series)
        for lag in lags:
            start_pos  = length_s + lag
            var_tau.append((np.var(np.subtract(np.roll(series, -lag)[-start_pos:-lag],series[-start_pos:-lag]))))
        
        # in case there are zeros replace with small value to be able log the values
        for j in range(len(var_tau)):
            if var_tau[j] == 0:
                var_tau[j] = 0.0001
        
        # calculate the hurst exponent
        hurst = np.polyfit(np.log(lags),np.log(var_tau), 1)[0]/2
        
        return hurst

    def calculate_half_life(self,x):
        """
        Function that calculates the half life parameter of a mean reversion series.
        
        :param x: (pd.Series) Time series to calculate the half life
        :return half_life : (float) Half life of series
        """
        
        # calculate lagged series and replace start with zero
        x_lag = np.roll(x, 1)
     
        # find the differences between oringal series and lagged
        x_dif = x - x_lag
        
        # adds intercept terms to x
        x_lag_constant = sm.add_constant(x_lag)

        # regress the returns differenced series and find the coefficient
        beta_coef = sm.OLS(x_dif[1:], x_lag_constant[1:]).fit().params[1]
        
        #check if it is zero to be able to divide
        if beta_coef == 0:
            half_life = 0
        else:
            half_life = -np.log(2) / beta_coef

        return half_life


    def spread_crossings(self,spread):
        """
        Function that counts the number of times that a spread crosses zero.
        
        :param spread: (pd.Series) Spread to calculate the number of crossings.
        :return spread_crossing : (int) The number of times that spread has been crossed.
        """
     
        spread_crossings = 0  
        
        # checks whether consecutive values have different signs
        for i in range(2,len(spread)-1):
            if ((spread[i] * spread[i+1]) < 0) or (spread[i] == 0):
                spread_crossings +=1

        return spread_crossings


    def pair_conditions_check(self,grouped_clusters,df_prices,hurst_threshold = 0.5,
                              min_half_life =5 ,max_half_life = 250, min_spread_crosses =12):
        """
        Fuction to calculate which pairs meet the Cointegration, Hurst exponent, Half life and
        Spread crossings conditions.
        
        :param grouped_clusters : (pandas DataFrame) Dataframe of the cluster of groups.
        :param df_prices : (pandas DataFrame)  Dataframe of historical stock prices.
        :param hurst_threshold : (int) Paramenter to specify when the Hurst condition is met. 
                                       Must be 0.5 for mean-reverting condition.
        :param min_half_life : (int) Minimum time for half life.
        :param max_half_life : (int) Maximum time for half life.
        :param min_spread_crosses : (int) Minimum number of spread crosses over the period.
        """
        
        conditions = ['Cointegration test','Hurst exponent test','Half life test',
                      'Spread crossings test','Conditions met','All conditions']
        
            
        # calculate number of pairs to evaluate
        pairs_numbers = np.sum(grouped_clusters['Count'].apply(lambda x :math.comb(x,2)))
        
        # initialize list of pairs
        pairs = []
        
        # calculate the list of all possible pair
        for i in range(grouped_clusters.shape[0]):
            for pair in it.combinations(grouped_clusters['Tickers'].iloc[i],2):
                pairs.append(pair)
        
        # dataframes to store conditions and spread of pairs
        conditions_met =  pd.DataFrame(0, index=pairs,columns=conditions)
        spreads = pd.DataFrame(0, index=df_prices.index,columns= pairs)

        for i, p in enumerate(pairs):
            
            # compute the log prices
            X = np.log(np.array(df_prices[p[0]]))
            Y = np.log(np.array(df_prices[p[1]]))
            
            # retrieve whether the stocks are cointegrated, and the spread between the stocks
            coint_bool, spread , dependant = self.coint_stats(X,Y)
            spreads[p] = spread         
            
            #checks cointegration condition
            if coint_bool:
                conditions_met['Cointegration test'].iloc[i] =  1 
            
            # calcluates hurst exponent and checks condition          
            hurst_e = self.hurst_exponent(spread) 
            if (hurst_e  < hurst_threshold) and (hurst_e != 0):
                conditions_met['Hurst exponent test'].iloc[i] =  1

            # calcluates half life and checks condition 
            half_life = self.calculate_half_life(spread)
            if (half_life >= min_half_life) & (max_half_life <= 250):
                conditions_met['Half life test'].iloc[i] =  1

            # calcluates spread crossings and checks condition 
            spread_cross = self.spread_crossings(spread)
            if spread_cross >= min_spread_crosses:
                conditions_met['Spread crossings test'].iloc[i] =  1

        # calculates the number of conditions met for each pair as
        # well as if all the conditions are satisfied
        conditions_met['Conditions met'] = conditions_met.sum(axis = 1)
        conditions_met['All conditions'] = conditions_met['Conditions met'].apply(lambda x :
                                                                                  1 if x == 4 else 0)
        # number of pair that satisfied each condition
        number_conditions = conditions_met.sum(axis = 0)[[0,1,2,3,5]]

        # list of pairs that met all the conditions
        possible_pairs  = conditions_met[conditions_met['All conditions'] == 1]
        possible_pairs = list(possible_pairs.index)


        return conditions_met , number_conditions, possible_pairs , spreads

    def plot_pairs(self,pairs_selected,df_prices,spreads):
        """
        Functions to plot the adjusted log prices of pairs and their spreads.
        
        :param pair_selected : (list) The list of pairs to plot.
        :param df_prices : (df_prices) Dataframe of prices.
        :param spreads : (pd.Series) Series of the spread between the pair of stoks.
        
        """
        for i in range(len(pairs_selected)):
            
            pair =  pairs_selected[i]
            df_prices[list(pair)]  
            
            # adjust the prices to display easily on graph
            means = np.log(df_prices[list(pair)].mean())
            series = np.log(df_prices[list(pair)]).sub(means)
            series['Spread'] = spreads[pair]
            title = list(pair)[0] + ' vs ' + list(pair)[1]
            
            # plot the graph
            fig = px.line(series,title = title).update_layout(
                yaxis_title="adjusted log Prices",legend_title="Series")
            
            fig.show()                        


 
        
        
            




        
        
    
  
    
  
 

        
       
   





















