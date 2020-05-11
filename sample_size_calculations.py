import numpy as np
import scipy.stats as scs

def ttest_sample_size(mu1, mu2, s1, s2, r, power,
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