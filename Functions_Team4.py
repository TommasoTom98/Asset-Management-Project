import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import the colormap
import seaborn as sns
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import datetime as datetime
from datetime import datetime
import time
from scipy.optimize import minimize, Bounds, LinearConstraint
from shapely.geometry import LineString
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

###########################################################################################
################################  FUNCTIONS ###############################################
###########################################################################################

def check_date(df1,df2,df3,df4,df5,names):
    """
    Function that check if all titles' date are the same
    """
    d=[0,-1]
    df1_d = df1.index[d]
    df2_d = df2.index[d]
    df3_d = df3.index[d]
    df4_d = df4.index[d]
    df5_d = df5.index[d]
        
    dictio = {"Start": [df1_d[0],df2_d[0],df3_d[0],df4_d[0],df5_d[0]],
              "End": [df1_d[-1],df2_d[-1],df3_d[-1],df4_d[-1],df5_d[-1]]}
    
    df_check = pd.DataFrame(dictio, index = names)
        
    cond = (df1_d == df2_d) & (df2_d == df3_d) &\
           (df3_d == df4_d) & (df4_d == df5_d)
    
    # First Datas check
    x = "First Datas are the same"
    y = "First same Data is: " + str(df_check["Start"].max())
    check_Start = np.where(df_check["Start"].nunique()==1,x,y)
    print(check_Start)
    print("")
    
    # Last Datas check
    x = "Last Datas are the same"
    y = "Last same Data is: " + str(df_check["End"].max()) 
    check_End = np.where(df_check["End"].nunique()==1,x,y)
    print(check_End)
    
    return df_check

#

def Graphs_Cov_Corr(df):
    """
    Function that plot:
    - Dataframe
    - Variance-Covariance Matrix
    - Correlation Matrix
    - HeatMap of Correlation Matriz
    """
    print("DataFrame: ")
    print(df.head())
    print("")
    # Variance - Covariance Matrix of LogReturn
    print("Variance - Covariance Matrix: ")
    print(df.cov())
    print("")
    print("Correlation Matrix: ")
    print(df.corr())
    print("")
    
    # HeatMap of Correlations
    plt.figure(figsize=(8, 6))  
    sns.heatmap(df.corr(), annot=False, fmt=".4f", cmap="RdBu", linewidths=0.5)
    # Plot correlation values inside cell
    for i in range(df.corr().shape[0]):
        for j in range(df.corr().shape[1]):
            plt.text(j + 0.5, i + 0.5, f"{df.corr().iloc[i, j]:.4f}", 
                     ha='center', va='center', color='white', fontsize=10)

def ShrinkMatrix(df,k):
    """
    Function that compute Shrinkage Matrix.
    Input:
    - Dataframe
    - k value
    """
    Cov_Matrix = df.cov()
    Corr_Matrix = df.corr()

    Mean_Corr = np.mean(Corr_Matrix.values[np.triu_indices_from(Corr_Matrix, k=1)])

    n = len(Corr_Matrix)
    Constant_Corr = np.full((n,n), Mean_Corr)
    np.fill_diagonal(Constant_Corr,1)

    Std_Matrix = np.sqrt(np.diag(Cov_Matrix))
    Constant_Cov = Constant_Corr * np.outer(Std_Matrix, Std_Matrix)

    Shrink_Matrix = k * Constant_Cov + (1-k) * Cov_Matrix

    return Shrink_Matrix

def ema_vector(df,lam):
    """
    Function that compute a vector of Exponential medium average.
    Input:
    - DataFrame
    - lambda
    """
    Ema = df.ewm(alpha=lam, adjust = True).mean().iloc[-1]
    return Ema
    
###########################################################################################
################################  METRICS #################################################
###########################################################################################

#########
# Title #
#########

def title_metrics(df):
    """
    Function that compute:
    - R_t: title Returns
    - std_t: title Standard Deviations
    """
    R_t = df.mean()
    Std_t = df.std()
    
    data_title = {"TotLogReturn": R_t,
                  "Std": Std_t}
    TitleMetrics = pd.DataFrame(data_title)
    return TitleMetrics

#############
# Portfolio #
#############

def Portfolio_metrics(weights, returns, df,Rf):
    """
    Function that compute:
    - R_p: title Returns
    - Std_p: title Standard Deviations
    - Sharpe_p: title Sharpe ratios
    """
    R_p = np.dot(weights, returns)
    
    CovMatrix = df.cov()
    Var_p = np.dot(weights.T, np.dot(CovMatrix,weights))
    Std_p = np.sqrt(Var_p)
    #
    Sharpe_p = (R_p-Rf)/Std_p
    
    data_port = {"P LogReturn": R_p,
                  "P Std": Std_p, 
                  "P Sharpe": Sharpe_p}
    columns = ["Portfolio"]
    PortMetrics = pd.DataFrame(data_port, columns)
    return PortMetrics

###########################################################################################
#############################  Efficient Frontier #########################################
###########################################################################################

def Efficient_Frontier(df, Names_t, Rf_percent):
    Start = datetime.now()
    print("Time start: " + str(Start))

    Rf = np.log(1 + Rf_percent) / 12  # Convert annual risk-free rate to monthly log rate
    R_title = df.mean()
    Std_t = df.std()
    Cov = df.cov()

    def portfolio_volatility(w, Cov):
        """Calculate portfolio volatility."""
        return np.sqrt(np.dot(w, np.dot(Cov, w)))

    def negative_sharpe_ratio(w, R_title, Cov, Rf):
        """Calculate negative Sharpe ratio."""
        port_return = np.dot(w, R_title)
        port_vol = portfolio_volatility(w, Cov)
        return -(port_return - Rf) / port_vol

    def calculate_frontier_and_cml(R_title, Cov, Rf):
        n_assets = len(R_title)
        x0 = np.ones(n_assets) / n_assets
        bounds = Bounds(0, 1)

        # Minimize portfolio volatility for MinVar portfolio
        res_min_var = minimize(portfolio_volatility, x0, args=(Cov,), method='SLSQP',
                               bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
        w_min_var = res_min_var.x
        ret_min_var = np.dot(w_min_var, R_title)
        vol_min_var = portfolio_volatility(w_min_var, Cov)

        # Minimize negative Sharpe ratio for Max Sharpe Portfolio
        res_sharpe = minimize(negative_sharpe_ratio, x0, args=(R_title, Cov, Rf), method='SLSQP',
                              bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
        w_sharpe = res_sharpe.x
        ret_sharpe = np.dot(w_sharpe, R_title)
        vol_sharpe = portfolio_volatility(w_sharpe, Cov)

        # Generate points for the efficient frontier and CML
        target_returns = np.linspace(ret_min_var, ret_sharpe, 50)
        vol_frontier = []
        for ret in target_returns:
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                           {'type': 'eq', 'fun': lambda w: np.dot(w, R_title) - ret}]
            res = minimize(portfolio_volatility, x0, args=(Cov,), method='SLSQP', bounds=bounds, constraints=constraints)
            vol_frontier.append(res.fun)

        # Calculate the CML
        slope = (ret_sharpe - Rf) / vol_sharpe
        vol_cml = np.linspace(0, max(vol_frontier), 100)
        ret_cml = Rf + slope * vol_cml

        return vol_frontier, target_returns, vol_cml, ret_cml, (w_min_var, ret_min_var, vol_min_var), (w_sharpe, ret_sharpe, vol_sharpe)

    # Execute the main calculation
    vol_frontier, target_returns, vol_cml, ret_cml, min_var_details, sharpe_details = calculate_frontier_and_cml(R_title, Cov, Rf)

    # Plotting
    plt.figure(figsize=(9, 5))
    colors = cm.rainbow(np.linspace(0, 1, len(Names_t)))  # Generate a color map
    plt.plot(vol_frontier, target_returns, 'g-', label='Efficient Frontier')
    plt.plot(vol_cml, ret_cml, 'r--', label='Capital Market Line (CML)')
    plt.scatter(min_var_details[2], min_var_details[1], color='red', s=100, label='Min Variance Portfolio')
    plt.scatter(sharpe_details[2], sharpe_details[1], color='blue', s=100, label='Max Sharpe Portfolio')
    for i, txt in enumerate(Names_t):
        plt.scatter(Std_t[i], R_title[i], color=colors[i], s=50)  # Use unique colors
        plt.annotate(txt, (Std_t[i], R_title[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Efficient Frontier with Individual Securities and CML')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Log Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    End = datetime.now()
    print("Time finish: " + str(End))
    print("Total time of execution: " + str(End - Start))

    return pd.DataFrame({
        'Portfolio': ['Min Variance', 'Max Sharpe'],
        'Weights': [np.round(min_var_details[0],2), np.round(sharpe_details[0],2)],
        'Volatility': [np.round(min_var_details[2],4), np.round(sharpe_details[2],4)],
        'Expected Return': [np.round(min_var_details[1],4), np.round(sharpe_details[1],4)]
    })

def Efficient_Frontier_Constraint(df, Names_t, Rf_percent, target):
    """
     Function that compute efficient frontier with different constraints and dataframe with
     MinVar and Tangent Portfolio.
     
     Output:
     - Graph without constraint
     - Graphs with constain
     - DataFrame with MinVar and Tangent Portfolios in each graph
     - DataFrame with datas about Portfolio Target
     
     Input:
     - Dataframe of title 
     - Title's names
     - Percent Risk Free
     - Target return
    """
    Start = datetime.now()
    print("Time start: "+ str(Start))

    Rf = np.log(1 + Rf_percent) / 12  # Monthly log risk-free 
    target = np.log(1 + target)

    R_title = df.mean()
    Std_t = df.std()
    Cov = df.cov()
  
    def portfolio_volatility(w, Cov):
        """
        Function that compute Portfolio volatility
        """
        return np.sqrt(np.dot(w, np.dot(Cov, w)))

    def negative_sharpe_ratio(w, R_title, Cov, Rf):
        """
        Function that compute negative sharpe ratio
        """
        port_return = np.dot(w, R_title)
        port_vol = portfolio_volatility(w, Cov)
        return -(port_return - Rf) / port_vol

    def calculate_frontier_and_cml(R_title, Cov, constraints_extra=None):
        """
        Function that compute:
        - Efficient frontier
        - MinVar Portfolio
        - CML
        - DataFrame with results
        """
        n_assets = len(R_title)
        x0 = np.ones(n_assets) / n_assets
        bounds = Bounds(0, 1)

        # MinVar portfolio (Minimizing portfolio volatility)
        res_min_var = minimize(portfolio_volatility, x0, args=(Cov,), method='SLSQP',
                               bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] +
                               (constraints_extra if constraints_extra else []))
        w_min_var = res_min_var.x
        ret_min_var = np.dot(w_min_var, R_title)
        vol_min_var = portfolio_volatility(w_min_var, Cov)

        # Max Sharp Portfolio (Minimizing negative Sharpe ratio)
        res_sharpe = minimize(negative_sharpe_ratio, x0, args=(R_title, Cov, Rf), method='SLSQP',
                              bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}] +
                              (constraints_extra if constraints_extra else []))
        w_sharpe = res_sharpe.x
        ret_sharpe = np.dot(w_sharpe, R_title)
        vol_sharpe = portfolio_volatility(w_sharpe, Cov)

        # Target returns generation between MinVar & Max Sharpe 
        target_returns = np.linspace(ret_min_var, ret_sharpe, 100)
        vol_frontier = []

        for target_ret in target_returns:
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                           {'type': 'eq', 'fun': lambda w: np.dot(w, R_title) - target_ret}]
            if constraints_extra: 
                constraints.extend(constraints_extra)
            res = minimize(portfolio_volatility, x0, args=(Cov,), method='SLSQP', bounds=bounds, constraints=constraints)
            vol_frontier.append(res.fun if res.success else np.nan)

        # CML
        slope = (ret_sharpe - Rf) / vol_sharpe
        vol_cml = np.linspace(0, max(vol_frontier), 100)
        ret_cml = Rf + slope * vol_cml

        return target_returns, vol_frontier, vol_cml, ret_cml, (w_min_var, vol_min_var, ret_min_var), (w_sharpe, vol_sharpe, ret_sharpe)
   
    # Constraint
    constraints_list = {
        'No Constraints': None,
        'Constraint 1': [{'type': 'eq', 'fun': lambda w: w[0] + w[1] - 0.5}],
        'Constraint 2': [{'type': 'ineq', 'fun': lambda w: w - 0.1}]
    }

    # Frontiers
    colors = ['green', 'blue', 'red']
    dfs = []
    intersections = []

    for (name, constraints), color, cml_color in zip(constraints_list.items(), colors, ['orange', 'purple', 'cyan']):

        target_returns, vol_frontier, vol_cml, ret_cml, min_var, sharpe = calculate_frontier_and_cml(R_title, Cov, constraints)

        # CML - Target line intersection
        cml_points = [(vol_cml[i], ret_cml[i]) for i in range(len(vol_cml))]
        target_line = LineString([(0, target), (max(vol_cml), target)])
        cml_line = LineString(cml_points)
        intersection = cml_line.intersection(target_line)

        vol_intersect, ret_intersect = (None, None)
        if not intersection.is_empty:
            vol_intersect, ret_intersect = intersection.x, intersection.y

        # Saving Results
        intersections.append({
            'Portfolio': name,
            'Intersection Volatility': round(vol_intersect,4),
            'Intersection LogReturn': round(ret_intersect,4)})

        ################### PLOT ###################

        plt.figure(figsize=(10, 6))
        plt.plot(vol_frontier, target_returns, label='Efficient Frontier', color=color)
        plt.plot(vol_cml, ret_cml, linestyle='--', color=cml_color, label='Capital Market Line (CML)')
        plt.scatter(min_var[1], min_var[2], color='red', s=100, label='Min Variance Portfolio')
        plt.scatter(sharpe[1], sharpe[2], color='blue', s=100, label='Max Sharpe Portfolio')

        for i, j in enumerate(Names_t):
            plt.scatter(Std_t[i], R_title[i], s=100)
            plt.text(Std_t[i], R_title[i], f"{j}", fontsize=8, ha='right')
        
        plt.axhline(y=target, color='grey', linestyle='--', label='Target Return')

        # Intersection
        if vol_intersect is not None and ret_intersect is not None:
            plt.scatter(vol_intersect, ret_intersect, color='magenta', s=100, label='CML-Target Intersection')

        plt.title(f'Efficient Frontier - {name}')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Saving Results
        df = pd.DataFrame({
            'Portfolio': [name],
            'MinVar Weights': [np.round(min_var[0], 2)],
            'MinVar Volatility': [round(min_var[1], 4)],
            'MinVar Log Return': [round(min_var[2], 4)],
            'Sharpe Weights':  [np.round(sharpe[0], 2)],
            'Sharpe Volatility': [round(sharpe[1], 4)],
            'Sharpe Log Return': [round(sharpe[2], 4)],})
        dfs.append(df)

    portfolio_df = pd.concat(dfs, ignore_index=True)
    intersection_df = pd.DataFrame(intersections)

    Rtan = portfolio_df["Sharpe Log Return"]
    
    # Weight of tangent portfolio for a target return
    # w_tangent_portfolio = (Rtarget - Risk Free) / (Rtangent_portfolio - Risk Free)
    # w_risk_free = 1 - w_tangent_portfolio
    w_tang = (target - Rf)/ (Rtan - Rf)

    intersection_df["wRtan"] = w_tang
    intersection_df["wRf"] = 1-w_tang

    # Check time of execution
    End = datetime.now()
    print("Time finsih: "+str(End))
    print("")
    print("Total time of execution: "+ str(End-Start))
    print("")

    #print(intersection_df)
    print(portfolio_df)
    print("")
    return intersection_df

###########################################################################################
#####################################  CAPM ###############################################
###########################################################################################

def CAPM (LOGRETURNTITOLI,LOGRETURNMKT,ERMKTLOG,RFLOG,MEDIETITOLI,NOMI,NOMIGRAFICO):
    """
    Function that computes: 
    - Title’s beta
    - Expected return 
    - Alpha 
    
    It also returns the SML graph where are also plotted also each title position and a 
    panda’s Data Frame where for each title are listed:
    - Company name
    - Beta
    - Average total monthly log return
    - CAPM expected return 
    - Alpha
    
    Input:
    - Titles log return 
    - Market log return
    - Expected market log return
    - Risk free log return
    - Expected titles log return
    - Titles name and market name
    - Titles name
    """
    Start = datetime.now()
    print("Time start: "+ str(Start))
    
    BETA = []
    RENDIMENTO_ATTESO = []
    ALFA = []
    
    for i in LOGRETURNTITOLI:
      cov_matrix = np.cov(i,LOGRETURNMKT )
      beta = cov_matrix[0, 1] / cov_matrix[1, 1]
      BETA.append(beta)
    for i in BETA:
      rendimento_atteso = RFLOG + i * (ERMKTLOG - RFLOG)
      RENDIMENTO_ATTESO.append(rendimento_atteso)
    for i in range(len(BETA)):
      alfa = MEDIETITOLI[i] - RENDIMENTO_ATTESO[i]
      ALFA.append(alfa)
    data = {'Company': NOMI,
          'Beta': [1]+ BETA,
          'Media Titoli': [ERMKTLOG] + MEDIETITOLI ,
          'Rendimento Atteso': [ERMKTLOG] + RENDIMENTO_ATTESO,
          'Alfa': [0]+ ALFA}
    df = pd.DataFrame(data)
    beta = np.linspace(0, 3, 100)
    ER_i = RFLOG + beta * (ERMKTLOG - RFLOG)
  
    # PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(beta, ER_i, label='Security Market Line (SML)', color='black')
    plt.scatter([0, 1], [RFLOG, ERMKTLOG], color='red')
    plt.text(0, RFLOG, "Rf", fontsize=12, ha='right')
    plt.text(1, ERMKTLOG, "MK", fontsize=12, ha='right')
    
    for i, (beta_val, er_val, nome) in enumerate(zip(BETA, MEDIETITOLI, NOMIGRAFICO)):
      plt.scatter(beta_val, er_val, color='blue')
      plt.text(beta_val, er_val, nome, fontsize=12, ha='right')

    plt.title("Security Market Line (SML)", fontsize=20)
    plt.xlabel("Beta (β)", fontsize=15)
    plt.ylabel("Expected Return E(R)", fontsize=15)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

    # Check time of execution
    End = datetime.now()
    print("Time finsih: "+str(End))
    print("")
    print("Total time of execution: "+ str(End-Start))
    print("")

    return df

###########################################################################################
################################ Black Litterman ##########################################
###########################################################################################

def Norm_Market_Weight(Stock_list):
    """
    Function that compute Normalized Market Weight.
    Input: list of stock
    """
    Stocks_MkCap = np.array([i["Company Market Cap"][-1] for i in Stock_list])
    Market_Cap_sum = np.sum(Stocks_MkCap)
    
    Stocks_Norm_MkCap = Stocks_MkCap / Market_Cap_sum
    
    return Stocks_Norm_MkCap

def lambda_Risk_aversion(market_return, rf_percent, market_var):
    """
    Function that compute Risk Aversion (lambda).
    Input:
    - Market Return
    - Risk Free rate (%)
    - Market Variance
    """
    # log market return
    market_return = np.log(1 + market_return)
    # log risk free
    risk_free_rate = np.log(1 + rf_percent) / 12 
    # log market variance
    market_variance = np.log(1 + market_var)
    
    # risk aversion ratio
    lambda_risk_aversion = (market_return - risk_free_rate) / market_variance

    return lambda_risk_aversion
    

def Mkt_Im_Ret(df, market_weights,lambda_risk_aversion):
    """
    Function that compute market implicit return 
    (Black Litterman framework)
    Input:
    - DataFrame with titles LogReturn
    - Market Weights
    - Risk Aversion (lambda)
    """
    covariance_matrix = df.cov()  
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2  # makes the matrix asymmetric
    implied_market_returns = lambda_risk_aversion * covariance_matrix.dot(market_weights)
    
    return implied_market_returns
    
def Mkt_Portfolio(market_weights):
    """
    Function that compute Market Portfolio, 
    assuming it is made up of few titles
    """
    market_portfolio = market_weights / np.sum(market_weights)
    #print("Market Portfolio Weights:",
    return market_portfolio

def Adj_Views(P, Q, tau, risk_aversion, implied_returns, market_portfolio, df):
    """
    Function that compute:
    - Adjusted Returns with Views
    - Adjusted Portfolio Weights with Views
    Input:
    - P: Views Matrix
    - Q: Views Vector
    - Tau: Model Views sensitivity
    - Risk Aversion
    - Implied Returns
    - Market Portfolio
    - DataFrame with titles LogReturn
    """
    covariance_matrix = df.cov()

    inv_tao_cov = np.linalg.inv(tau * covariance_matrix)
    M = np.linalg.inv(inv_tao_cov + P.T @ P)
    adjusted_returns = M @ (inv_tao_cov @ implied_returns + P.T @ Q)
    adjusted_weights = (1 / risk_aversion) * np.linalg.inv(covariance_matrix) @ adjusted_returns

    # Plotting Adjusted weights vs Market Portfolio
    plt.figure(figsize=(6, 4))
    bar_width = 0.35
    index = np.arange(len(market_portfolio))
    plt.bar(index, market_portfolio, bar_width, label='Market Weights')
    plt.bar(index + bar_width, adjusted_weights, bar_width, label='Adjusted Weights')

    plt.xlabel('Assets')
    plt.ylabel('Weights')
    plt.title('Adjusted Weights vs Market Weights')
    plt.xticks(index + bar_width / 2, df.columns, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Printing Results
    print("Market Weights Portfolio:")
    print(market_portfolio)
    print("")
    print("Implied Market Returns:")
    print(implied_returns)
    print("")
    print("Adjusted Returns with Views:")
    print(adjusted_returns)
    print("")
    print("Adjusted Weights with Views:")
    print(adjusted_weights)

def TEV_Views(n_assets, P, Q, benchmark_weights,risk_aversion,implied_returns,df,tau):
    """
    Function that compute Views incorporation  based on the Tracking Error Variance frontier.
    Input:
    - Number of assets
    - Benchmark Weights
    - Risk aversion (lambda)
    - Implied Market Returns
    - DataFrame with titles LogReturns
    - Tau: Model Views sensitivity
    """
    covariance_matrix = df.cov()
    inv_tao_cov = np.linalg.inv(tau * covariance_matrix)
    M = np.linalg.inv(inv_tao_cov + P.T @ P)
    adjusted_returns = M @ (inv_tao_cov @ implied_returns + P.T @ Q)
    
    def tracking_error_variance(portfolio_weights, benchmark_weights, covariance_matrix):
        """
        Function that compute Tracking Error Variance
        """
        difference = portfolio_weights - benchmark_weights
        return difference.T @ covariance_matrix @ difference
    
    def objective(weights):
        """
        Objective fucntion used to minimize TEV
        """
        return tracking_error_variance(weights, benchmark_weights, covariance_matrix)

    # Sum of Weights = 1
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    # Weights values have a range [-1,1]
    bounds = [(-1, 1) for _ in range(n_assets)]

    # Optimization problem
    initial_guess = benchmark_weights  # Start value
    result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds)
    optimized_weights = result.x

    #Plotting Optimazed Weights
    plt.figure(figsize=(6, 4))
    plt.bar(df.columns, optimized_weights)
    plt.title('Optimized Portfolio Weights for Minimizing TEV')
    plt.xlabel('Assets')
    plt.ylabel('Weights')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
    
    # Printing Results
    print("Updated Returns with Views:", adjusted_returns)
    print("")
    print("Optimised Weights for Minimizing TEV:", optimized_weights)