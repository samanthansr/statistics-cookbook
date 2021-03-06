import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as scs

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
        
    
    def z_test_statistic(self, alternative):

        delta = self.p_B - self.p_A
        pooled_proportion = (self.converted_A + self.converted_B) / (self.n_A + self.n_B)
        std_error_pooled = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/self.n_A + 1/self.n_B))
        
        z_score = (delta - 0) / std_error_pooled
        
        if alternative == 'smaller':
            pval = scs.norm.cdf(z_score)
        elif alternative == 'larger':
            pval = 1 - scs.norm.cdf(z_score)
        elif alternative == 'two-sided':
            pval = (1 - scs.norm.cdf(abs(z_score))) * 2
            
        return (z_score, pval, std_error_pooled)
        
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
            # it's always cv - z_score because: 
                # if Z > 0 (+ve), then we're looking at a right-tail test, and Z beta would be e.g. 1.96 - 2.03
                # if Z < 0 (-ve), then it's a left-tail test, and Z beta would be -1.96 - (-2.03)
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
        
    def confidence_intervals(self, sig_level, alternative):
        
        # if the null hypothesis (0, because no difference) 
        # does not lie within the confidence interval, 
        # we have sufficient evidence to reject the null hypothesis
        
        delta = self.p_B - self.p_A
        pooled_proportion = (self.converted_A + self.converted_B) / (self.n_A + self.n_B)
        std_error_pooled = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/self.n_A + 1/self.n_B))
        
        if alternative == 'two-sided':
            z_value = abs(scs.norm(0, 1).ppf(1-(sig_level/2)))
            margin_of_error = z_value * std_error_pooled
            confidence_interval = (delta - margin_of_error, delta + margin_of_error)
        
        elif alternative == 'larger':
            z_value = scs.norm(0.1).ppf(1 - sig_level)
            margin_of_error = z_value * std_error_pooled
            confidence_interval = (delta + margin_of_error, np.inf)
            
        elif alternative == 'smaller':
            z_value = scs.norm(0.1).ppf(sig_level)
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

        diff_mean = self.X_bar_B - self.X_bar_A
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
        diff_mean = self.X_bar_B - self.X_bar_A
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

        # ncp calculations taken from NCSS formula
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Two-Sample_T-Tests_Assuming_Equal_Variance.pdf
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Two-Sample_T-Tests_Allowing_Unequal_Variance.pdf

        if pooled_var:
            self.dof = self.n_A + self.n_B - 2
            self.pooled_variance = (sum_squared_difference_X_A + sum_squared_difference_X_B) / self.dof
            self.std_error = np.sqrt(self.pooled_variance * (1 / self.n_A + 1 / self.n_B))
            self.ncp = (self.X_bar_B - self.X_bar_A) / (np.sqrt(self.pooled_variance * (1/self.n_A + 1/self.n_B)))

        else: 
            self.dof = (self.var_A/self.n_A + self.var_B/self.n_B) ** 2
            self.dof /= (
                (((self.var_A/self.n_A) ** 2) / (self.n_A - 1)) +
                (((self.var_B/self.n_B) ** 2) / (self.n_B - 1))
            )
            self.std_error = np.sqrt(self.var_A / self.n_A + self.var_B / self.n_B)
            self.ncp = (self.X_bar_B - self.X_bar_A) / np.sqrt(self.var_A/self.n_A + self.var_B/self.n_B)

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
        # not sure if this should use non central t-distribution 
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

        diff_mean = self.X_bar_B - self.X_bar_A

        # use self.variance / pooled variance

        t_null = scs.t(df=self.dof, loc=0, scale=self.std_error) # 0 because H0 is no difference
        t_alt = scs.nct(df=self.dof, nc=self.ncp, loc=0, scale=self.std_error)

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
        
        diff_mean = self.X_bar_B - self.X_bar_A
        
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
        
    def power(self, sig_level, alternative):
        
        # implementation of power calculation taken from NCSS:
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Two-Sample_T-Tests_Assuming_Equal_Variance.pdf
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Two-Sample_T-Tests_Allowing_Unequal_Variance.pdf 

        t_alt = scs.nct(df=self.dof, nc=self.ncp)
        cv = self.critical_value(sig_level, alternative)
        
        if alternative == 'smaller':
            return t_alt.cdf(cv)
        
        elif alternative == 'larger':
            return 1 - t_alt.cdf(cv)
        
        elif alternative == 'two-sided':
            return [t_alt.cdf(cv[0]), 1 - t_alt.cdf(cv[1])]
        
    def beta(self, sig_level, alternative):

        t_alt = scs.nct(df=self.dof, nc=self.ncp)
        cv = self.critical_value(sig_level, alternative)
        
        power = self.power(sig_level, alternative)
        
        if alternative == 'two-sided':
            return t_alt.cdf(cv[1]) - t_alt.cdf(cv[0])
            
        else:
            return 1 - power
    
    def get_test_results(self, sig_level, alternative):
        
        t_score, pval, std_error = self.t_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        results = {
            't Score': t_score,
            'p-value': pval,
            'Std. Error': std_error,
            'Degrees of Freedom': self.dof,
            'Critical Value (t-alpha)': critical_value,
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
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        ### Plotting Null and Alternate Hypothesis ### 
        
        fig, ax = plt.subplots(figsize=(12,6))

        t_null = scs.t(df=self.dof, loc=0, scale=1)
        t_alt = scs.nct(df=self.dof, nc=self.ncp)

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

    def confidence_intervals(self, sig_level, alternative):
        
        # if the null hypothesis (0, because no difference) 
        # does not lie within the confidence interval, 
        # we have sufficient evidence to reject the null hypothesis
        
        diff_mean = self.X_bar_B - self.X_bar_A
        t_null = scs.t(df=self.dof, loc=0, scale=1)
        
        if alternative == 'two-sided':
            t_value = abs(t_null.ppf(sig_level/2))
            margin_of_error = t_value * self.std_error
            confidence_interval = (diff_mean - margin_of_error, diff_mean + margin_of_error)
        
        elif alternative == 'larger':
            t_value = t_null.ppf(1 - sig_level)
            margin_of_error = t_value * self.std_error
            confidence_interval = (diff_mean + margin_of_error, np.inf)
            
        elif alternative == 'smaller':
            t_value = t_null.ppf(sig_level)
            margin_of_error = t_value * self.std_error
            confidence_interval = (-np.inf, diff_mean + margin_of_error)

        return confidence_interval

class TwoSamplePairedTTest:

    def __init__(self, X_A, X_B):

        self.X_A = X_A
        self.X_B = X_B

        self.n_A = len(X_A)
        self.n_B = len(X_B)
        assert self.n_A == self.n_B

        self.diff = np.array(X_A) - np.array(X_B)
        self.diff_mean = np.mean(self.diff) # mean of the differences
        self.diff_sum_squared_difference = sum((self.diff - self.diff_mean) ** 2)
        self.diff_var = self.diff_sum_squared_difference / (self.n_A - 1)
        self.diff_stddev = np.sqrt(self.diff_var) # stddev of the differences

        self.dof = self.n_A - 1
        self.std_error = self.diff_stddev / np.sqrt(self.n_A)

        self.t_value = (self.diff_mean - 0) / self.std_error
        
    def plot_samples(self):
        
        fig = plt.figure(figsize=(12, 6))
        
        sns.distplot(self.X_A, label='X_A / Before', color='cornflowerblue')
        sns.distplot(self.X_B, label='X_B / After', color='lightcoral')
        
        plt.title('Samples Distribution of Values (Histogram)')
        plt.ylabel('Frequencies')
        
        plt.legend()
        plt.show()

    def plot_sampling_distribution_of_difference_in_means(self):

        t_null = scs.t(df=self.dof, loc=0, scale=self.std_error) # 0 because H0 is no difference
        t_alt = scs.nct(df=self.dof, nc=self.t_value, loc=0, scale=self.std_error)

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
        ax.axvline(x=self.diff_mean, linestyle='--', c='lightcoral')
        
        plt.xlabel('Distribution of the Difference in Sample Means')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()

    def t_test_statistic(self, alternative):
        
        # diff_mean = self.X_bar_B - self.X_bar_A
        
        # t_score = (self.diff_mean - 0) / self.std_error
        
        if alternative == 'smaller':
            pval = scs.t.cdf(self.t_value, self.dof)
        elif alternative == 'larger':
            pval = 1 - scs.t.cdf(self.t_value, self.dof)
        elif alternative == 'two-sided':
            pval = (1 - scs.t.cdf(abs(self.t_value), self.dof)) * 2

        return (self.t_value, pval, self.std_error)
        
    def critical_value(self, sig_level, alternative):

        t_null = scs.t(df=self.dof, loc=0, scale=1)
        
        if alternative == 'smaller':
            return t_null.ppf(sig_level)
        
        elif alternative == 'larger':
            return t_null.ppf(1 - sig_level)
        
        elif alternative == 'two-sided':
            return [t_null.ppf(sig_level / 2), t_null.ppf(1 - (sig_level / 2))]
        
    def power(self, sig_level, alternative):
        
        # implementation of power calculation taken from NCSS:
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Multiple_Testing_for_One_Mean-One-Sample_or_Paired_Data.pdf

        t_alt = scs.nct(df=self.dof, nc=self.t_value)
        cv = self.critical_value(sig_level, alternative)
        
        if alternative == 'smaller':
            return t_alt.cdf(cv)
        
        elif alternative == 'larger':
            return 1 - t_alt.cdf(cv)
        
        elif alternative == 'two-sided':
            return [t_alt.cdf(cv[0]), 1 - t_alt.cdf(cv[1])]
        
    def beta(self, sig_level, alternative):

        t_alt = scs.nct(df=self.dof, nc=self.t_value)
        cv = self.critical_value(sig_level, alternative)
        
        power = self.power(sig_level, alternative)
        
        if alternative == 'two-sided':
            return t_alt.cdf(cv[1]) - t_alt.cdf(cv[0])
            
        else:
            return 1 - power
    
    def get_test_results(self, sig_level, alternative):
        
        t_score, pval, std_error = self.t_test_statistic(alternative=alternative)
        critical_value = self.critical_value(sig_level, alternative)
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        results = {
            't Score': t_score,
            'p-value': pval,
            'Std. Error': std_error,
            'Degrees of Freedom': self.dof,
            'Critical Value (t-alpha)': critical_value,
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
        power = self.power(sig_level, alternative)
        beta = self.beta(sig_level, alternative)
        
        ### Plotting Null and Alternate Hypothesis ### 
        
        fig, ax = plt.subplots(figsize=(12,6))

        t_null = scs.t(df=self.dof, loc=0, scale=1)
        t_alt = scs.nct(df=self.dof, nc=self.t_value)

        x = np.linspace(
            min(t_null.ppf(0.001), t_alt.ppf(0.01), -abs(t_score)),
            max(t_null.ppf(0.99), t_alt.ppf(0.99), abs(t_score)),
            1000
        )

        y_null = t_null.pdf(x)
        ax.plot(x, y_null, label='control/null', c='cornflowerblue', linewidth=3)

        if alternative == 'two-sided':
            ax.vlines(x=[-t_score, t_score], ymin=0, ymax=0.4, linestyle='--', color='cornflowerblue', linewidth=2)
            ax.text(abs(t_score), 0.40, '$t$' + ' = {:.5f}'.format(t_score), 
                    bbox={'facecolor':'cornflowerblue', 'alpha':0.5}, horizontalalignment='center')
            ax.text(-abs(t_score), 0.40, '$t$' + ' = {:.5f}'.format(-abs(t_score)), 
                    bbox={'facecolor':'cornflowerblue', 'alpha':0.5}, horizontalalignment='center')
        else: 
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
                ax.text(-abs(t_score), 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
                ax.text(abs(t_score), 0.05,'p-value = {:.5f}'.format(pval/2), 
                        style='italic', bbox={'facecolor':'cornflowerblue', 
                                              'alpha':0.25})
            
            elif alternative == 'smaller':
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x < t_score))
                ax.text(t_score, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                        bbox={'facecolor':'cornflowerblue', 
                              'alpha':0.25})
                
            else:
                ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x > t_score))
                ax.text(t_score, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
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

    def confidence_intervals(self, sig_level, alternative):
        
        # if the null hypothesis (0, because no difference) 
        # does not lie within the confidence interval, 
        # we have sufficient evidence to reject the null hypothesis
        
        diff_mean = self.diff_mean
        t_null = scs.t(df=self.dof, loc=0, scale=1)
        
        if alternative == 'two-sided':
            t_value = abs(t_null.ppf(sig_level/2))
            margin_of_error = t_value * self.std_error
            confidence_interval = (diff_mean - margin_of_error, diff_mean + margin_of_error)
        
        elif alternative == 'larger':
            t_value = t_null.ppf(1 - sig_level)
            margin_of_error = t_value * self.std_error
            confidence_interval = (diff_mean + margin_of_error, np.inf)
            
        elif alternative == 'smaller':
            t_value = t_null.ppf(sig_level)
            margin_of_error = t_value * self.std_error
            confidence_interval = (-np.inf, diff_mean + margin_of_error)

        return confidence_interval

class Chi2Ind:

    def __init__(self, obs_freq):
        self.obs_freq = obs_freq
        self.dof = (obs_freq.shape[0]-1) * (obs_freq.shape[1]-1)
        self.n = obs_freq.sum()
    
    def expected_freq(self):

        row_sums = self.obs_freq.sum(axis=1)
        col_sums = self.obs_freq.sum(axis=0)
        
        expected_freq = []
        total_counts = self.obs_freq.sum()

        for row_count in row_sums:
            row_exp_freq = []

            for col_count in col_sums:
                
                counts = col_count * row_count / total_counts
                row_exp_freq.append(counts)
            
            expected_freq.append(row_exp_freq)

        return np.array(expected_freq)
    
    def cramers_v(self):
        """
        Calculates statistical strength using Cramer's V

        References: 
            - https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
            - http://www.real-statistics.com/chi-square-and-f-distributions/effect-size-chi-square/
        """

        chisq_statistic, pval, _ = self.chisq_statistic()

        v = np.sqrt((chisq_statistic/self.n)/(min(len(self.obs_freq[0]), 
                                                  len(self.obs_freq)) - 1))

        return v

    def chisq_statistic(self):

        # implementing only a right-tailed test because chi-square is always a one-sided test
        # reference: https://stats.stackexchange.com/questions/22347/is-chi-squared-always-a-one-sided-test
        
        exp_freq = self.expected_freq()
        residuals = self.obs_freq - exp_freq
        chi2_pts = (residuals ** 2) / exp_freq
        chi2_stat = chi2_pts.sum()

        pval = 1 - scs.chi2.cdf(chi2_stat, self.dof)

        return (chi2_stat, pval, self.dof)

    def critical_value(self, sig_level):

        chi2_null = scs.chi2(df=self.dof, loc=0, scale=1)
        
        return chi2_null.ppf(1 - sig_level)
        
    def power(self, sig_level):
        
        # implementation of power calculation taken from NCSS (p.3 and 4 of pdf):
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Chi-Square_Tests.pdf

        chi2_stat, _, _ = self.chisq_statistic()
        chi2_alt = scs.ncx2(df=self.dof, nc=chi2_stat)
        
        cv = self.critical_value(sig_level)

        return 1 - chi2_alt.cdf(cv)
       
        
    def beta(self, sig_level):
        
        power = self.power(sig_level)
        
        return 1 - power
    
    def get_test_results(self, sig_level):
        
        chi2_stat, pval, _ = self.chisq_statistic()
        critical_value = self.critical_value(sig_level)
        power = self.power(sig_level)
        beta = self.beta(sig_level)
        v = self.cramers_v()
        
        results = {
            'Chi2 Statistic': chi2_stat,
            'p-value': pval,
            'Degrees of Freedom': self.dof,
            'Effect Size (Cramer\'s V)': v,
            'Critical Value (chi2-alpha)': critical_value,
            'Power': power,
            'Beta (Type II Error)': beta
        }
        
        return results

    def plot_sampling_dist_of_difference_standardized(self, sig_level, 
                                                      show_alpha=False, show_pvalue=False,
                                                      show_beta=False, show_power=False):
        
        ### calculating values ### 
        chi2_stat, pval, _ = self.chisq_statistic()
        critical_value = self.critical_value(sig_level)
        power = self.power(sig_level)
        beta = self.beta(sig_level)
        
        ### Plotting Null and Alternate Hypothesis ### 
        
        fig, ax = plt.subplots(figsize=(12,6))

        chi2_null = scs.chi2(df=self.dof, loc=0, scale=1)
        chi_alt = scs.ncx2(df=self.dof, nc=chi2_stat)

        x = np.linspace(
            # min(chi2_null.ppf(0.001), chi_alt.ppf(0.01), -abs(chi2_stat)),
            0.001,
            max(chi2_null.ppf(0.99), chi_alt.ppf(0.99), abs(chi2_stat)),
            1000
        )

        y_null = chi2_null.pdf(x)
        y_alt = chi_alt.pdf(x)

        max_pdf = max(max(y_null), max(y_alt)) # for position of label

        ax.plot(x, y_null, label='null', c='cornflowerblue', linewidth=3)
        ax.plot(x, y_alt, label='alternate', c='lightcoral', linestyle=':')

        ax.axvline(x=chi2_stat, linestyle='--', c='cornflowerblue', linewidth=2)
        ax.text(chi2_stat, max_pdf, '$chi2$' + ' = {:.5f}'.format(chi2_stat), 
                bbox={'facecolor':'cornflowerblue', 'alpha':0.5}, horizontalalignment='left')


        ### Plotting critical regions ### 
        ax.axvline(x=critical_value, linestyle = '--', c='black')
        ax.text(critical_value, max_pdf - 0.01, '$t_{\\alpha}$' + ' = {:.5f}'.format(critical_value), 
                bbox={'facecolor':'white', 'alpha':0.5}, horizontalalignment='right')
            
        ### Plotting shading areas ### 
            
        if show_pvalue:
            ### SHADING IN P-VALUE ###
            ax.fill_between(x, 0, y_null, color='cornflowerblue', alpha=0.25, where=(x > chi2_stat))
            ax.text(chi2_stat, 0.05, 'p-value = {:.5f}'.format(pval), style='italic', 
                    bbox={'facecolor':'cornflowerblue', 
                            'alpha':0.25})
        
        if show_alpha:
            ### SHADING IN ALPHA/SIG. LEVEL ###

            ax.fill_between(x, 0, y_null, color='grey', alpha=0.25, where=(x > critical_value))
            ax.text(critical_value, 0, 
                    '$\\alpha = {:.5f}$'.format(1 - chi2_null.cdf(critical_value)), 
                    bbox={'facecolor':'grey', 'alpha':0.25}, horizontalalignment='center')
        
        if show_power:
            ### SHADING IN POWER ###
            ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x > critical_value))
            ax.text(2.8, 0.1, '$1 - \\beta$' + ' = {:.5f}'.format(power), style='italic', 
                    bbox={'facecolor':'lightcoral', 
                            'alpha':0.25})
                
        if show_beta:
            ### SHADING IN BETA (TYPE II ERROR) ###
            ax.fill_between(x, 0, y_alt, color='lightcoral', alpha=0.25, where=(x < critical_value))
            ax.text(0, 0.25, '$\\beta$' + ' = {:.5f}'.format(beta), style='italic', 
                    bbox={'facecolor':'lightcoral', 
                            'alpha':0.25})
        
        plt.xlabel('chi2-value, relative to the NULL hypothesis')
        plt.ylabel('Probability Density Function (PDF)')
        plt.legend()
        plt.show()
