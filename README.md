# Statistics Cookbook I

This project was started in an attempt to apply what I learned while while going through Khan Academy's AP Statistics. While doing up the section on hypothesis tests, I realised the course did not cover power and sample size calculations and so I did some further exploration on those topics.

The notebook covers the following topics:
- Binomial Distributions
- Central Limit Theorem: Simulation showing how the sampling distribution approximates a normal distribution as sample size increases, even if the population does not follow a normal distribution
- Hypothesis Test: 2 Sample Z Test of Proportions, 2 Sample Z Test of Means, 2 Sample Independent t-test, 2 Sample Paired t-test, Chi Square Test

For each test, I explored the methods available in either `scipy` or `statsmodels` to calculate p-value, power and sample size. At the same time, to better understand the concepts and how calculations were done, I implemented it from scratch in both `statistical_tests.py` and `sample_size_calculations.py`