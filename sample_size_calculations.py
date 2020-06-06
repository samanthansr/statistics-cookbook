import numpy as np
import scipy.stats as scs

def ttest_ind_sample_size(mu1, mu2, s1, s2, r, power,
                          sig_level=0.05, alternative='two-sided', pooled=True):
    
    n1 = 2 # initialisation
    n2 = n1 * r
    sim_power = 0
    
    while sim_power < power:
        
        n1 += 1
        n2 = n1 * r
    
        if pooled:

            dof = n1 + n2 - 2        
            pooled_var = (((s1 ** 2) * (n1-1)) + ((s2 ** 2) * (n2-1))) / dof
            std_error = np.sqrt(pooled_var * (1/n1 + 1/n2))

        else:

            var1 = (s1 ** 2) # assuming unbiased sample standard deviation
            var2 = (s2 ** 2)

            dof = (var1/n1 + var2/n2) ** 2
            dof /= (
                (((var1/n1) ** 2) / (n1 - 1)) +
                (((var2/n2) ** 2) / (n2-1))
            )
            std_error = np.sqrt(var1/n1 + var2/n2)
        
        ncp = (mu2 - mu1) / std_error

        t_null = scs.t(df=dof, loc=0, scale=1)
        t_alt = scs.nct(df=dof, nc=ncp)

        if alternative == 'smaller':
            cv = t_null.ppf(sig_level)
            sim_power = t_alt.cdf(cv)

        elif alternative == 'larger':
            cv = t_null.ppf(1 - sig_level)
            sim_power = 1 - t_alt.cdf(cv)

        elif alternative == 'two-sided':
            cv = [t_null.ppf(sig_level / 2), t_null.ppf(1 - (sig_level / 2))]
            sim_power = sum([t_alt.cdf(cv[0]), 1 - t_alt.cdf(cv[1])])

    print('ncp: ', ncp)
    print('Critical t: ', cv)
    print('Actual Power: ', sim_power)
    
    return (np.ceil(n1), np.ceil(n2))

def ttest_paired_sample_size(starting_n, effect_size, 
                             power=0.8, sig_level=0.05, alternative='two-sided'):

    n = starting_n # initialisation
    sim_power = 0
    
    while sim_power < power:
        
        n += 1
    
        dof = n - 1
        ncp = effect_size * np.sqrt(n)

        t_null = scs.t(df=dof, loc=0, scale=1)
        t_alt = scs.nct(df=dof, nc=ncp)

        if alternative == 'smaller':
            cv = t_null.ppf(sig_level)
            sim_power = t_alt.cdf(cv)

        elif alternative == 'larger':
            cv = t_null.ppf(1 - sig_level)
            sim_power = 1 - t_alt.cdf(cv)

        elif alternative == 'two-sided':
            cv = [t_null.ppf(sig_level / 2), t_null.ppf(1 - (sig_level / 2))]
            sim_power = sum([t_alt.cdf(cv[0]), 1 - t_alt.cdf(cv[1])])

    print('ncp: ', ncp)
    print('Critical t: ', cv)
    print('Actual Power: ', sim_power)

    return np.ceil(n)

def effect_size_paired_ttest_from_differences(diff_mean, diff_sd):
    
    es = diff_mean / diff_sd
    return es

def effect_size_paired_ttest_from_group_params(mu1, mu2, s1, s2, corr):
        
    diff_mean = mu2 - mu1
    diff_sd = np.sqrt(s1**2 + s2**2 - (2*corr*s1*s2))
    es = diff_mean / diff_sd

    return es

def chi2ind_sample_size(starting_n, effect_size, dof,
                        power=0.8, sig_level=0.05):
    """
    Tested against reference in NCSS PASS manual: 
    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Chi-Square_Tests.pdf
    (Page 7)
    """

    n = starting_n # initialisation
    sim_power = 0
    
    while sim_power < power:
        
        n += 1
    
        ncp = (effect_size ** 2) * n

        chi2_null = scs.chi2(df=dof, loc=0, scale=1)
        chi2_alt = scs.ncx2(df=dof, nc=ncp)

        cv = chi2_null.ppf(1 - sig_level)
        sim_power = 1 - chi2_alt.cdf(cv)

    print('ncp: ', ncp)
    print('Critical chi2: ', cv)
    print('Actual Power: ', sim_power)

    return np.ceil(n)