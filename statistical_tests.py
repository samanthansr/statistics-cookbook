import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as scs
from statsmodels.stats.proportion import proportions_ztest

import warnings

class TwoPropZTest:
    
    def __init__(self, df=None, 
                 n_A=None, converted_A=None, 
                 n_B=None, converted_B=None):
        
        if (df is None) & (n_A is None 
                           and converted_A is None 
                           and n_B is None 
                           and converted_B is None):
            raise Exception('Input either a df or parameters for n_A, converted_A, n_B and converted_B')
            
        elif (df is None) & (n_A is None 
                             or converted_A is None 
                             or n_B is None 
                             or converted_B is None):
            raise Exception('Input all fields for n_A, converted_A, n_B, converted_B')
        
        elif df is not None:
            self.raw_data = df
            self.group_label = input('Group Label')
            self.group_names = re.sub('(?<=,)[\s]', '', input('Group Names in Array')).split(',')
            self.result_label = input('Result Label')
            
            self.n_A = len(df[df[self.group_label]==self.group_names[0]])
            self.converted_A = len(df[(df[self.group_label]==self.group_names[0]) & 
                                      (df[self.result_label]==1)])
            self.p_A = self.converted_A / self.n_A
            
            self.n_B = len(df[df[self.group_label]==self.group_names[1]])
            self.converted_B = len(df[(df[self.group_label]==self.group_names[1]) & 
                                      (df[self.result_label]==1)])
            self.p_B = self.converted_B / self.n_B
            
        elif df is None:
            
            self.n_A = n_A
            self.converted_A = converted_A
            self.p_A = converted_A / n_A
            
            self.n_B = n_B
            self.converted_B = converted_B
            self.p_B = converted_B / n_B
            
    def contingency_table(self):
        pivot = pd.pivot_table(self.raw_data, 
                               index=self.group_label, 
                               aggfunc = {self.result_label: [np.sum,
                                                              lambda x: len(x),
                                                              np.mean]})

        # dropping redundant level 'converted'
        pivot.columns = pivot.columns.droplevel(0)

        # rename columns
        pivot.rename(columns={'<lambda>': 'total',
                              'mean': 'rate', 
                              'sum': 'converted'}, inplace=True)


        return pivot[['converted', 'total', 'rate']]
    
    def plot_binomial_distribution(self):
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        x_A = np.linspace(start=0, stop=self.n_A, num=self.n_A+1)
        y_A = scs.binom(self.n_A, self.p_A).pmf(x_A)

        x_B = np.linspace(start=0, stop=self.n_B, num=self.n_B+1)
        y_B = scs.binom(self.n_B, self.p_B).pmf(x_B)

        ax.bar(x_A, y_A, alpha=0.5, label='control/null', color='cornflowerblue')
        ax.bar(x_B, y_B, alpha=0.5, label='experiment/alternate', color='lightcoral')

        plt.xlabel('# of Successes')
        plt.ylabel('Probability Mass Function (PMF)')
        plt.legend()
        plt.show()
    
    def plot_sampling_dist(self):
        
        SE_A = np.sqrt(self.p_A * (1 - self.p_A) / self.n_A)
        SE_B = np.sqrt(self.p_B * (1 - self.p_B) / self.n_B)
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        x = np.linspace(0, 1, 100)
        
        y_A = scs.norm(self.p_A, SE_A).pdf(x)
        ax.plot(x, y_A, label='control/null', c='cornflowerblue')
        ax.axvline(x=self.p_A, linestyle='--', c='cornflowerblue')
        
        y_B = scs.norm(self.p_B, SE_B).pdf(x)
        ax.plot(x, y_B, label='experiment/alternate', c='lightcoral')
        ax.axvline(x=self.p_B, linestyle='--', c='lightcoral')
        
        plt.xlabel('Sample Proportion')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

    def plot_sampling_dist_of_difference(self):
        
        delta = self.p_B - self.p_A
        pooled_proportion = (self.converted_A + self.converted_B) / (self.n_A + self.n_B)
        std_error_pooled = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/self.n_A + 1/self.n_B))
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        x = np.linspace(-1, 1, 100)
        
        y_null = scs.norm(0, std_error_pooled).pdf(x)   # 0 because null hypothesis is no difference
        ax.plot(x, y_null, label='control/null', c='cornflowerblue')
        ax.axvline(x=0, linestyle='--', c='lightgrey')
        
        y_alt = scs.norm(delta, std_error_pooled).pdf(x)
        ax.plot(x, y_alt, label='experiment/alternate', c='lightcoral')
        ax.axvline(x=delta, linestyle='--', c='lightcoral')
        
        plt.xlabel('Difference in Sample Proportions')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()
        
    
    def z_test_statistic(self, alternative, method='scipy', check_equality=False):
        
        # statsmodels
        count = np.array([self.converted_B, self.converted_A])
        nobs = np.array([self.n_B, self.n_A])
        value = 0 
        z_score, pval = proportions_ztest(count, nobs, value, 
                                          alternative = alternative,
                                          prop_var = False)
        
        # manual
        delta = self.p_B - self.p_A
        pooled_proportion = (self.converted_A + self.converted_B) / (self.n_A + self.n_B)
        std_error_pooled = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/self.n_A + 1/self.n_B))
        
        z_score_manual = (delta - 0) / std_error_pooled
        
        if alternative == 'smaller':
            pval_manual = scs.norm.cdf(z_score_manual)
        elif alternative == 'larger':
            pval_manual = 1 - scs.norm.cdf(z_score_manual)
        elif alternative == 'two-sided':
            pval_manual = (1 - scs.norm.cdf(abs(z_score_manual))) * 2
        
        if check_equality:
            assert z_score_manual == z_score
            assert pval_manual == pval
            
        if method == 'scipy':
            return (z_score, pval, std_error_pooled)
        elif method == 'manual':
            return (z_score_manual, pval_manual, std_error_pooled)
        
    def critical_value(self, sig_level, alternative):
        
        if alternative == 'smaller':
            return scs.norm(0, 1).ppf(sig_level)
        
        elif alternative == 'larger':
            return scs.norm(0, 1).ppf(1 - sig_level)
        
        elif alternative == 'two-sided':
            return [scs.norm(0, 1).ppf(sig_level / 2), scs.norm(0, 1).ppf(1 - (sig_level / 2))]
        
    def z_beta(self, sig_level, alternative, z_score_method='scipy'):
        
        z_score, pval, std_error = self.z_test_statistic(alternative=alternative, method=z_score_method)
        critical_value = self.critical_value(sig_level, alternative)
        
        if alternative == 'two-sided':
            z_beta = [critical_value[0] - z_score, critical_value[1] - z_score]
            
        else:
            z_beta = critical_value - z_score

        return z_beta
        
    def power(self, sig_level, alternative, z_score_method='scipy'):
        
        z_beta = self.z_beta(sig_level, alternative, z_score_method)
        
        if alternative == 'smaller':
            return scs.norm.cdf(z_beta)
        
        elif alternative == 'larger':
            return 1 - scs.norm.cdf(z_beta)
        
        elif alternative == 'two-sided':
            return [scs.norm.cdf(z_beta[0]), 1 - scs.norm.cdf(z_beta[1])]
        
    def beta(self, sig_level, alternative, z_score_method='scipy'):
        
        power = self.power(sig_level, alternative, z_score_method)
        z_beta = self.z_beta(sig_level, alternative, z_score_method)
        
        if alternative == 'two-sided':
            return scs.norm.cdf(z_beta[1]) - scs.norm.cdf(z_beta[0])
            
        else:
            return 1 - power
    
    def get_test_results(self, sig_level, alternative, z_score_method='scipy'):
        
        z_score, pval, std_error = self.z_test_statistic(alternative=alternative, method=z_score_method)
        critical_value = self.critical_value(sig_level, alternative)
        z_beta = self.z_beta(sig_level, alternative, z_score_method)
        power = self.power(sig_level, alternative, z_score_method)
        beta = self.beta(sig_level, alternative, z_score_method)
        
        results = {
            'Z Score': z_score,
            'p-value': pval,
            'Std. Error': std_error,
            'Critical Value (Z-alpha)': critical_value,
            'Z-Beta': z_beta,
            'Power': power,
            'Beta (Type II Error)': beta
        }
        
        return results
        
    def plot_sampling_dist_of_difference_standardized(self, sig_level, alternative, 
                                                      show_alpha=False, show_pvalue=False,
                                                      show_beta=False, show_power=False):
        
        ### calculating values ### 
        z_score, pval, _ = self.z_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        z_beta = self.z_beta(sig_level, alternative, z_score_method='scipy')
        power = self.power(sig_level, alternative, z_score_method='scipy')
        beta = self.beta(sig_level, alternative, z_score_method='scipy')
        
        ### Plotting Null and Alternate Hypothesis ### 
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        x = np.linspace(min(0, z_score) - 4, max(0, z_score) + 4, 1000)
        
        y_null = scs.norm(0, 1).pdf(x)
        ax.plot(x, y_null, label='control/null', c='cornflowerblue', linewidth=3)
        ax.axvline(x=z_score, linestyle='--', c='cornflowerblue', linewidth=2)
        ax.text(z_score, 0.40, '$Z$' + ' = {:.5f}'.format(z_score), 
                bbox={'facecolor':'cornflowerblue', 'alpha':0.5}, horizontalalignment='center')
        
        y_alt = scs.norm(z_score, 1).pdf(x)
        ax.plot(x, y_alt, label='experiment/alternate', c='lightcoral', linestyle=':')
        
        ### Plotting critical regions ### 
        if alternative == 'two-sided':
            ax.axvline(x=critical_value[0], linestyle = '--', c='black')
            ax.axvline(x=critical_value[1], linestyle = '--', c='black')
            ax.text(critical_value[0], 0.40, '$Z_{\\alpha}$' + ' = {:.5f}'.format(critical_value[0]), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
            ax.text(critical_value[1], 0.40, '$Z_{\\alpha}$' + ' = {:.5f}'.format(critical_value[1]), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
        else: 
            ax.axvline(x=critical_value, linestyle = '--', c='black')
            ax.text(critical_value, 0.40, '$Z_{\\alpha}$' + ' = {:.5f}'.format(critical_value), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
            
        ### Plotting shading areas ### 
            
        if show_pvalue:
            ### SHADING IN P-VALUE ###
            if alternative == 'two-sided':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(abs(x) > abs(z_score)))
                ax.text(-1.5, 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
                ax.text(1.5, 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
            
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x < z_score))
                ax.text(-1.5, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
                
            else:
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x > z_score))
                ax.text(1.5, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
        
        if show_alpha:
            ### SHADING IN ALPHA/SIG. LEVEL ###
            if alternative == 'two-sided':
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(abs(x) > critical_value[1]))
                ax.text(critical_value[0], 0, 
                        '$\\alpha = {:.5f}$'.format(scs.norm.cdf(critical_value[0])), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                ax.text(critical_value[1], 0, 
                        '$\\alpha = {:.5f}$'.format(scs.norm.cdf(critical_value[0])), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x < critical_value))
                ax.text(critical_value, 0, 
                    '$\\alpha = {:.5f}$'.format(scs.norm.cdf(critical_value)), 
                    bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                
            else:
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x > critical_value))
                ax.text(critical_value, 0, 
                        '$\\alpha = {:.5f}$'.format(1 - scs.norm.cdf(critical_value)), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
            
        
        if show_power:
            ### SHADING IN POWER ###
            if alternative == 'smaller':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
                ax.text(-2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            elif alternative == 'larger':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
                ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            else:
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(abs(x) > critical_value[1]))
                ax.text(-2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power[0]), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power[1]), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                ax.text(2.7, 0.3, 'Total Stat. Power: {:.5f}'.format(sum(power)), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                
        
        if show_beta:
            ### SHADING IN BETA (TYPE II ERROR) ###
            
            if alternative == 'smaller':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            elif alternative == 'larger':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                
            else: 
                
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(abs(x) < critical_value[1]))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
        
        
        plt.xlabel('Z-value, relative to the NULL hypothesis')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()
        
    def confidence_intervals(self, confidence_level, alternative):
        
        # if the null hypothesis (0, because no difference) 
        # does not lie within the confidence interval, 
        # we have sufficient evidence to reject the null hypothesis
        
        delta = self.p_B - self.p_A
        pooled_proportion = (self.converted_A + self.converted_B) / (self.n_A + self.n_B)
        std_error_pooled = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/self.n_A + 1/self.n_B))
        
        if alternative == 'two-sided':
            z_value = abs(scs.norm(0, 1).ppf((1-confidence_level)/2))
            margin_of_error = z_value * std_error_pooled
            confidence_interval = (delta - margin_of_error, delta + margin_of_error)
        
        elif alternative == 'larger':
            z_value = scs.norm(0.1).ppf(1 - confidence_level)
            margin_of_error = z_value * std_error_pooled
            confidence_interval = (delta + margin_of_error, np.inf)
            
        elif alternative == 'smaller':
            z_value = scs.norm(0.1).ppf(confidence_level)
            margin_of_error = z_value * std_error_pooled
            confidence_interval = (-np.inf, delta + margin_of_error)

        return confidence_interval


class TwoSampleZTest:
    
    def __init__(self, popvar_A, popvar_B, X_A=None, X_B=None, n_A=None, X_bar_A=None, n_B=None, X_bar_B=None):
        
        if (X_A is None and 
            X_B is None and
            n_A is None and 
            X_bar_A is None and 
            n_B is None and 
            X_bar_B is None):
            raise Exception('Input either 2 samples in X_A and X_B as numpy arrays'
                            ' or parameters for n_A, X_bar_A, n_B, X_bar_B')
            
        elif (X_A is None and X_B is None) and (n_A is None or
                                                X_bar_A is None or
                                                n_B is None or
                                                X_bar_B is None):
            raise Exception('Input all fields for n_A, X_bar_A, n_B, X_bar_B')
            
        elif (X_A is not None and 
              X_B is not None and
              n_A is not None and 
              X_bar_A is not None and 
              n_B is not None and 
              X_bar_B is not None):
            
            assert len(X_A) == n_A, ('X_A array length != n_A. Either make sure both values are alligned or '
                                     'input only the array or n_A and X_bar_A.')
            assert len(X_B) == n_B, ('X_B array length != n_B. Either make sure both values are alligned or '
                                     'input only the array or n_B and X_bar_B.')
            assert X_A.mean() == X_bar_A, ('X_A mean != X_bar_A. Either make sure both values are alligned or '
                                        'input only the array or n_A and X_bar_A.')
            assert X_B.mean() == X_bar_B, ('X_B mean != X_bar_B. Either make sure both values are alligned or '
                                        'input only the array or n_B and X_bar_B.')
            
            self.n_A = n_A
            self.X_bar_A = X_bar_A
            self.n_B = n_B
            self.X_bar_B = X_bar_B
            
        elif (X_A is not None and X_B is not None):
            self.n_A = len(X_A)
            self.X_bar_A = X_A.mean()
            self.n_B = len(X_B)
            self.X_bar_B = X_B.mean()
            self.X_A = X_A
            self.X_B = X_B
            
        elif (n_A is not None and 
              X_bar_A is not None and 
              n_B is not None and 
              X_bar_B is not None):
            self.n_A = n_A
            self.X_bar_A = X_bar_A
            self.n_B = n_B
            self.X_bar_B = X_bar_B
            
        self.popvar_A = popvar_A
        self.popvar_B = popvar_B

        # sample variance = population variance / N
        self.sigma_sq_A = self.popvar_A / self.n_A
        self.sigma_sq_B = self.popvar_B / self.n_B

        self.sigma_A = np.sqrt(self.sigma_sq_A)
        self.sigma_B = np.sqrt(self.sigma_sq_B)
        
    def plot_samples(self):
        
        fig = plt.figure(figsize=(12, 6))
        
        sns.distplot(self.X_A, label='X_A / Control', color='cornflowerblue')
        sns.distplot(self.X_B, label='X_B / Experiment', color='lightcoral')
        
        plt.title('Samples Distribution of Values (Histogram)')
        plt.ylabel('Frequencies')
        
        plt.legend()
        plt.show()

    def plot_sampling_distribution(self):

        fig, ax = plt.subplots(figsize=(12,6))

        x = np.linspace(
            min((self.X_bar_A - 3 * self.sigma_A), (self.X_bar_B - 3 * self.sigma_B)), 
            max((self.X_bar_A + 3 * self.sigma_A), (self.X_bar_B + 3 * self.sigma_B)),
            1000
            )

        y_A = scs.norm(self.X_bar_A, self.sigma_A).pdf(x)
        y_B = scs.norm(self.X_bar_B, self.sigma_B).pdf(x)

        ax.plot(x, y_A, label='X_A / Control', color='cornflowerblue')
        ax.axvline(x=self.X_bar_A, linestyle='--', c='cornflowerblue')

        ax.plot(x, y_B, label='X_B / Experiment', color='lightcoral')
        ax.axvline(x=self.X_bar_B, linestyle='--', c='lightcoral')

        plt.title('Sampling Distribution of the Sampling Mean')
        plt.xlabel('Sample Means')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

    def plot_sampling_distribution_of_difference_in_means(self):

        diff_mean = self.X_bar_A - self.X_bar_B
        diff_var = (self.popvar_A / self.n_A) + (self.popvar_B / self.n_B)
        diff_std = np.sqrt(diff_var)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.linspace(min((0 - 3 * diff_std), (diff_mean - 3 * diff_std)),
                        max((0 + 3 * diff_std), (diff_mean + 3 * diff_std)),
                        1000)

        y_null = scs.norm(0, diff_std).pdf(x) # 0 because H0 is no difference
        ax.plot(x, y_null, label='Null Hypothesis', c='cornflowerblue')
        ax.axvline(x=0, linestyle='--', c='lightgrey')

        y_alt = scs.norm(diff_mean, diff_std).pdf(x)
        ax.plot(x, y_alt, label='Alternate Hypothesis', c='lightcoral')
        ax.axvline(x=diff_mean, linestyle='--', c='lightcoral')
        
        plt.xlabel('Distribution of the Difference in Sample Means')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

    def z_test_statistic(self, alternative):
        
        # manual
        diff_mean = self.X_bar_A - self.X_bar_B
        diff_var = (self.popvar_A / self.n_A) + (self.popvar_B / self.n_B)
        diff_std = np.sqrt(diff_var)
        
        z_score = (diff_mean - 0) / diff_std
        
        if alternative == 'smaller':
            pval = scs.norm.cdf(z_score)
        elif alternative == 'larger':
            pval = 1 - scs.norm.cdf(z_score)
        elif alternative == 'two-sided':
            pval = (1 - scs.norm.cdf(abs(z_score))) * 2

        return (z_score, pval, diff_std)
        
    def critical_value(self, sig_level, alternative):
        
        if alternative == 'smaller':
            return scs.norm(0, 1).ppf(sig_level)
        
        elif alternative == 'larger':
            return scs.norm(0, 1).ppf(1 - sig_level)
        
        elif alternative == 'two-sided':
            return [scs.norm(0, 1).ppf(sig_level / 2), scs.norm(0, 1).ppf(1 - (sig_level / 2))]
        
    def z_beta(self, sig_level, alternative):
        
        z_score, pval, std_error = self.z_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        
        if alternative == 'two-sided':
            z_beta = [critical_value[0] - z_score, critical_value[1] - z_score]
            
        else:
            z_beta = critical_value - z_score

        return z_beta
        
    def power(self, sig_level, alternative):
        
        z_beta = self.z_beta(sig_level, alternative)
        
        if alternative == 'smaller':
            return scs.norm.cdf(z_beta)
        
        elif alternative == 'larger':
            return 1 - scs.norm.cdf(z_beta)
        
        elif alternative == 'two-sided':
            return [scs.norm.cdf(z_beta[0]), 1 - scs.norm.cdf(z_beta[1])]
        
    def beta(self, sig_level, alternative):
        
        power = self.power(sig_level, alternative)
        z_beta = self.z_beta(sig_level, alternative)
        
        if alternative == 'two-sided':
            return scs.norm.cdf(z_beta[1]) - scs.norm.cdf(z_beta[0])
            
        else:
            return 1 - power
    
    def get_test_results(self, sig_level, alternative):
        
        z_score, pval, std_error = self.z_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        z_beta = self.z_beta(sig_level, alternative)
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        results = {
            'Z Score': z_score,
            'p-value': pval,
            'Std. Error': std_error,
            'Critical Value (Z-alpha)': critical_value,
            'Z-Beta': z_beta,
            'Power': power,
            'Beta (Type II Error)': beta
        }
        
        return results
        
    def plot_sampling_dist_of_difference_standardized(self, sig_level, alternative, 
                                                      show_alpha=False, show_pvalue=False,
                                                      show_beta=False, show_power=False):
        
        ### calculating values ### 
        z_score, pval, _ = self.z_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        z_beta = self.z_beta(sig_level, alternative)
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        ### Plotting Null and Alternate Hypothesis ### 
        
        fig, ax = plt.subplots(figsize=(12,6))
        
        x = np.linspace(min(0, z_score) - 4, max(0, z_score) + 4, 1000)
        
        y_null = scs.norm(0, 1).pdf(x)
        ax.plot(x, y_null, label='control/null', c='cornflowerblue', linewidth=3)
        ax.axvline(x=z_score, linestyle='--', c='cornflowerblue', linewidth=2)
        ax.text(z_score, 0.40, '$Z$' + ' = {:.5f}'.format(z_score), 
                bbox={'facecolor':'cornflowerblue', 'alpha':0.5}, horizontalalignment='center')
        
        y_alt = scs.norm(z_score, 1).pdf(x)
        ax.plot(x, y_alt, label='experiment/alternate', c='lightcoral', linestyle=':')
        
        ### Plotting critical regions ### 
        if alternative == 'two-sided':
            ax.axvline(x=critical_value[0], linestyle = '--', c='black')
            ax.axvline(x=critical_value[1], linestyle = '--', c='black')
            ax.text(critical_value[0], 0.40, '$Z_{\\alpha}$' + ' = {:.5f}'.format(critical_value[0]), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
            ax.text(critical_value[1], 0.40, '$Z_{\\alpha}$' + ' = {:.5f}'.format(critical_value[1]), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
        else: 
            ax.axvline(x=critical_value, linestyle = '--', c='black')
            ax.text(critical_value, 0.40, '$Z_{\\alpha}$' + ' = {:.5f}'.format(critical_value), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
            
        ### Plotting shading areas ### 
            
        if show_pvalue:
            ### SHADING IN P-VALUE ###
            if alternative == 'two-sided':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(abs(x) > abs(z_score)))
                ax.text(-1.5, 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
                ax.text(1.5, 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
            
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x < z_score))
                ax.text(-1.5, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
                
            else:
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x > z_score))
                ax.text(1.5, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
        
        if show_alpha:
            ### SHADING IN ALPHA/SIG. LEVEL ###
            if alternative == 'two-sided':
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(abs(x) > critical_value[1]))
                ax.text(critical_value[0], 0, 
                        '$\\alpha = {:.5f}$'.format(scs.norm.cdf(critical_value[0])), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                ax.text(critical_value[1], 0, 
                        '$\\alpha = {:.5f}$'.format(scs.norm.cdf(critical_value[0])), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x < critical_value))
                ax.text(critical_value, 0, 
                    '$\\alpha = {:.5f}$'.format(scs.norm.cdf(critical_value)), 
                    bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                
            else:
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x > critical_value))
                ax.text(critical_value, 0, 
                        '$\\alpha = {:.5f}$'.format(1 - scs.norm.cdf(critical_value)), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
            
        
        if show_power:
            ### SHADING IN POWER ###
            if alternative == 'smaller':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
                ax.text(-2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            elif alternative == 'larger':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
                ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            else:
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(abs(x) > critical_value[1]))
                ax.text(-2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power[0]), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power[1]), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                ax.text(2.7, 0.3, 'Total Stat. Power: {:.5f}'.format(sum(power)), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                
        
        if show_beta:
            ### SHADING IN BETA (TYPE II ERROR) ###
            
            if alternative == 'smaller':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            elif alternative == 'larger':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                
            else: 
                
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(abs(x) < critical_value[1]))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
        
        
        plt.xlabel('Z-value, relative to the NULL hypothesis')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

class TwoSampleIndTTest:
    
    def __init__(self, X_A, X_B, pooled_var):

        self.X_A = np.array(X_A)
        self.X_B = np.array(X_B)

        self.X_bar_A = np.array(self.X_A).mean()
        self.X_bar_B = np.array(self.X_B).mean()

        self.n_A = len(self.X_A)
        self.n_B = len(self.X_B)

        sum_squared_difference_X_A = sum((self.X_A - self.X_bar_A) ** 2)
        sum_squared_difference_X_B = sum((self.X_B - self.X_bar_B) ** 2)

        # unbiased sample variance
        self.var_A = sum_squared_difference_X_A / (self.n_A - 1)
        self.var_B = sum_squared_difference_X_B / (self.n_B - 1)

        self.sigma_A = np.sqrt(self.var_A)
        self.sigma_B = np.sqrt(self.var_B)

        if pooled_var and max(self.sigma_A, self.sigma_B) > 2 * min(self.sigma_A, self.sigma_B):
            warnings.warn('Larger standard deviation is more than 2x larger than the smaller standard deviation: '
                          'sigma_A = {:.2f}, sigma_B = {:.2f}'.format(self.sigma_A, self.sigma_B))

        if pooled_var:
            self.dof = self.n_A + self.n_B - 2
            self.pooled_variance = (sum_squared_difference_X_A + sum_squared_difference_X_B) / self.dof
            self.std_error = np.sqrt(self.pooled_variance * (1 / self.n_A + 1 / self.n_B))

        else: 
            self.dof = (self.var_A/self.n_A + self.var_B/self.n_B) ** 2
            self.dof /= (
                (((self.var_A/self.n_A) ** 2) / (self.n_A - 1)) +
                (((self.var_B/self.n_B) ** 2) / (self.n_B - 1))
            )
            self.std_error = np.sqrt(self.var_A / self.n_A + self.var_B / self.n_B)

            dof = (var_A/n_A + var_B/n_B) ** 2
            dof /= (
                (((var_A/n_A) ** 2) / (n_A - 1)) +
                (((var_B/n_B) ** 2) / (n_B - 1))
            )

    def plot_samples(self):
        
        fig = plt.figure(figsize=(12, 6))
        
        sns.distplot(self.X_A, label='X_A / Control', color='cornflowerblue')
        sns.distplot(self.X_B, label='X_B / Experiment', color='lightcoral')
        
        plt.title('Samples Distribution of Values (Histogram)')
        plt.ylabel('Frequencies')
        
        plt.legend()
        plt.show()

    def plot_sampling_distribution(self):

        fig, ax = plt.subplots(figsize=(12,6))

        t_A = scs.t(df=self.n_A - 1, loc=self.X_bar_A, scale=self.sigma_A)
        t_B = scs.t(df=self.n_B - 1, loc=self.X_bar_B, scale=self.sigma_B)

        x = np.linspace(
            min(t_A.ppf(0.01), t_B.ppf(0.01)),
            max(t_A.ppf(0.99), t_B.ppf(0.99)),
            1000
        )

        y_A = t_A.pdf(x)
        y_B = t_B.pdf(x)

        ax.plot(x, y_A, label='X_A / Control', color='cornflowerblue')
        ax.axvline(x=self.X_bar_A, linestyle='--', c='cornflowerblue')

        ax.plot(x, y_B, label='X_B / Experiment', color='lightcoral')
        ax.axvline(x=self.X_bar_B, linestyle='--', c='lightcoral')

        plt.title('Sampling Distribution of the Sampling Mean')
        plt.xlabel('Sample Means')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

    def plot_sampling_distribution_of_difference_in_means(self):

        diff_mean = self.X_bar_A - self.X_bar_B

        # use self.variance / pooled variance

        t_null = scs.t(df=self.dof, loc=0, scale=self.std_error) # 0 because H0 is no difference
        t_alt = scs.t(df=self.dof, loc=diff_mean, scale=self.std_error)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.linspace(
            min(t_null.ppf(0.01), t_alt.ppf(0.01)),
            max(t_null.ppf(0.99), t_alt.ppf(0.99)),
            1000
        )

        y_null = t_null.pdf(x)
        ax.plot(x, y_null, label='Null Hypothesis', c='cornflowerblue')
        ax.axvline(x=0, linestyle='--', c='lightgrey')

        y_alt = t_alt.pdf(x)
        ax.plot(x, y_alt, label='Alternate Hypothesis', c='lightcoral')
        ax.axvline(x=diff_mean, linestyle='--', c='lightcoral')
        
        plt.xlabel('Distribution of the Difference in Sample Means')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

    def t_test_statistic(self, alternative):
        
        diff_mean = self.X_bar_A - self.X_bar_B
        
        t_score = (diff_mean - 0) / self.std_error
        
        if alternative == 'smaller':
            pval = scs.t.cdf(t_score, self.dof)
        elif alternative == 'larger':
            pval = 1 - scs.t.cdf(t_score, self.dof)
        elif alternative == 'two-sided':
            pval = (1 - scs.t.cdf(abs(t_score), self.dof)) * 2

        return (t_score, pval, self.std_error)
        
    def critical_value(self, sig_level, alternative):

        t_null = scs.t(df=self.dof, loc=0, scale=1)
        
        if alternative == 'smaller':
            return t_null.ppf(sig_level)
        
        elif alternative == 'larger':
            return t_null.ppf(1 - sig_level)
        
        elif alternative == 'two-sided':
            return [t_null.ppf(sig_level / 2), t_null.ppf(1 - (sig_level / 2))]
        
    def t_beta(self, sig_level, alternative):
        
        t_score, pval, std_error = self.t_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        
        if alternative == 'two-sided':
            t_beta = [critical_value[0] - t_score, critical_value[1] - t_score]
            
        else:
            t_beta = critical_value - t_score

        return t_beta

    # def z_beta(self, sig_level, alternative, z_score_method='scipy'):
        
    #     z_score, pval, std_error = self.z_test_statistic(alternative=alternative, method=z_score_method)
    #     critical_value = self.critical_value(sig_level, alternative)
        
    #     if alternative == 'two-sided':
    #         z_beta = [critical_value[0] - z_score, critical_value[1] - z_score]
            
    #     else:
    #         z_beta = critical_value - z_score

    #     return z_beta
        
    # def power(self, sig_level, alternative, z_score_method='scipy'):
        
    #     z_beta = self.z_beta(sig_level, alternative, z_score_method)
        
    #     if alternative == 'smaller':
    #         return scs.norm.cdf(z_beta)
        
    #     elif alternative == 'larger':
    #         return 1 - scs.norm.cdf(z_beta)
        
    #     elif alternative == 'two-sided':
    #         return [scs.norm.cdf(z_beta[0]), 1 - scs.norm.cdf(z_beta[1])]
        
    def power(self, sig_level, alternative):
        
        t_null = scs.t(df=self.dof, loc=0, scale=1)
        t_beta = self.t_beta(sig_level, alternative)
        
        if alternative == 'smaller':
            return t_null.cdf(t_beta)
        
        elif alternative == 'larger':
            return 1 - t_null.cdf(t_beta)
        
        elif alternative == 'two-sided':
            # return [scs.norm.cdf(t_beta[0]), 1 - scs.norm.cdf(t_beta[1])]
            return [t_null.cdf(t_beta[0]), 1 - t_null.cdf(t_beta[1])]
            # return [t_alt.cdf(-t_score), 1 - t_alt.cdf(t_score)]
            # need to amend t-score, right now hardcoded for current case
        
    def beta(self, sig_level, alternative):

        t_score, _, __ = self.t_test_statistic(alternative=alternative)
        t_alt = scs.t(df=self.dof, loc=t_score, scale=1)
        
        power = self.power(sig_level, alternative)
        t_beta = self.t_beta(sig_level, alternative)
        
        if alternative == 'two-sided':
            return t_alt.cdf(t_beta[1]) - t_alt.cdf(t_beta[0])
            
        else:
            return 1 - power
    
    def get_test_results(self, sig_level, alternative):
        
        t_score, pval, std_error = self.t_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        t_beta = self.t_beta(sig_level, alternative)
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        results = {
            't Score': t_score,
            'p-value': pval,
            'Std. Error': std_error,
            'Critical Value (t-alpha)': critical_value,
            't-Beta': t_beta,
            'Power': power,
            'Beta (Type II Error)': beta
        }
        
        return results

    def plot_sampling_dist_of_difference_standardized(self, sig_level, alternative, 
                                                      show_alpha=False, show_pvalue=False,
                                                      show_beta=False, show_power=False):
        
        ### calculating values ### 
        t_score, pval, _ = self.t_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        t_beta = self.t_beta(sig_level, alternative)
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        ### Plotting Null and Alternate Hypothesis ### 
        
        fig, ax = plt.subplots(figsize=(12,6))

        t_null = scs.t(df=self.dof, loc=0, scale=1)
        t_alt = scs.t(df=self.dof, loc=t_score, scale=1)

        x = np.linspace(
            min(t_null.ppf(0.01), t_alt.ppf(0.01)),
            max(t_null.ppf(0.99), t_alt.ppf(0.99)),
            1000
        )

        
        y_null = t_null.pdf(x)
        ax.plot(x, y_null, label='control/null', c='cornflowerblue', linewidth=3)
        ax.axvline(x=t_score, linestyle='--', c='cornflowerblue', linewidth=2)
        ax.text(t_score, 0.40, '$t$' + ' = {:.5f}'.format(t_score), 
                bbox={'facecolor':'cornflowerblue', 'alpha':0.5}, horizontalalignment='center')
        
        y_alt = t_alt.pdf(x)
        ax.plot(x, y_alt, label='experiment/alternate', c='lightcoral', linestyle=':')
        
        ### Plotting critical regions ### 
        if alternative == 'two-sided':
            ax.axvline(x=critical_value[0], linestyle = '--', c='black')
            ax.axvline(x=critical_value[1], linestyle = '--', c='black')
            ax.text(critical_value[0], 0.40, '$t_{\\alpha}$' + ' = {:.5f}'.format(critical_value[0]), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
            ax.text(critical_value[1], 0.40, '$t_{\\alpha}$' + ' = {:.5f}'.format(critical_value[1]), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
        else: 
            ax.axvline(x=critical_value, linestyle = '--', c='black')
            ax.text(critical_value, 0.40, '$t_{\\alpha}$' + ' = {:.5f}'.format(critical_value), 
                    bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='center')
            
        ### Plotting shading areas ### 
            
        if show_pvalue:
            ### SHADING IN P-VALUE ###
            if alternative == 'two-sided':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(abs(x) > abs(t_score)))
                ax.text(-1.5, 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
                ax.text(1.5, 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
            
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x < t_score))
                ax.text(-1.5, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
                
            else:
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x > t_score))
                ax.text(1.5, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
        
        if show_alpha:
            ### SHADING IN ALPHA/SIG. LEVEL ###
            if alternative == 'two-sided':
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(abs(x) > critical_value[1]))
                ax.text(critical_value[0], 0, 
                        '$\\alpha = {:.5f}$'.format(t_null.cdf(critical_value[0])), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                ax.text(critical_value[1], 0, 
                        '$\\alpha = {:.5f}$'.format(t_null.cdf(critical_value[0])), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x < critical_value))
                ax.text(critical_value, 0, 
                    '$\\alpha = {:.5f}$'.format(t_null.cdf(critical_value)), 
                    bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
                
            else:
                ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x > critical_value))
                ax.text(critical_value, 0, 
                        '$\\alpha = {:.5f}$'.format(1 - t_null.cdf(critical_value)), 
                        bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
            
        
        if show_power:
            ### SHADING IN POWER ###
            if alternative == 'smaller':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
                ax.text(-2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            elif alternative == 'larger':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
                ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            else:
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(abs(x) > critical_value[1]))
                ax.text(-2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power[0]), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power[1]), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                ax.text(2.7, 0.3, 'Total Stat. Power: {:.5f}'.format(sum(power)), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                
        
        if show_beta:
            ### SHADING IN BETA (TYPE II ERROR) ###
            
            if alternative == 'smaller':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
            elif alternative == 'larger':
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
                
            else: 
                
                ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(abs(x) < critical_value[1]))
                ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                        bbox={'facecolor':'lightcoral', 
                              'alpha':0.25})
        
        
        plt.xlabel('t-value, relative to the NULL hypothesis')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()