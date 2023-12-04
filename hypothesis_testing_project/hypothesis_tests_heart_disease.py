#The purpose of this project is tp take heart disease research data (from UC Irvine Machine Learning Repo) and answer a collection of
# questions by running hypothesis tests for the association between two variables. Some questions that are answered are:

# Do people with heart disease have high cholesterol levels on average?
# Is the proportion of patients in the data with high fasting blood sugar larger than the national average?
# Is there a significant difference in cholesterol levels between heart disease/no heart disease patients?
# Do patients with different chest pain types have signifcantly different thalach (defined below)?
# Is the presence of heart disease significantly associated with chest pain type?

# To answer these questions, I use the 1-sample t-test, binomial test, 2-sample t-test, ANOVA test, Tukey range test, and chi-squared test.

# importing required libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import binom_test
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

from ucimlrepo import fetch_ucirepo


# fetch dataset from UC Irvine Machine Learning Repo.
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets
heart = pd.concat([X, y], axis=1)
# print(heart.info())

# Variable descriptions taken from the data source (https://archive.ics.uci.edu/dataset/45/heart+disease)
# age: age in years
# sex: sex assigned at birth; 'male' or 'female'
# trestbps: resting blood pressure in mm Hg
# chol: serum cholesterol in mg/dl
# cp: chest pain type
# exang: patient experiences exercise-induced angina (1: yes; 0: no)
# fbs: patient’s fasting blood sugar is >120 mg/dl (1: yes; 0: no)
# thalach: maximum heart rate achieved in exercise test
#num: heart disease type

# heart_disease: whether the patient is found to have heart disease. Derived from the num column.
heart['heart_disease'] = np.where(heart.num == 0, 'Absence', 'Presence')
# print(heart.head())


### In the first part of this project, I investigate the fbs (fasting blood sugar) and chol (cholesterol level) variables. 

# Cholesterol levels of 240 or higher are considered "High". Do people with heart disease have high cholesterol levels on average?
# Null Hypothesis: People with heart disease have an average cholesterol level equal to 240 mg/dl.
# Alternative Hypothesis: People with heart disease have an average cholesterol level that is greater than 240 mg/dl.

#splitting the data into patients with/without heart disease.
yes_hd = heart[heart.heart_disease == 'Presence']
no_hd = heart[heart.heart_disease == 'Absence']

#examine heart disease subset
print(yes_hd.head())

#pull chol levels from heart disease subset into variable and calc mean
chol_hd = yes_hd['chol']
mean_chol_hd = np.mean(chol_hd)
print("The mean cholesterol levels of the patients with heart disease: ", mean_chol_hd)

high_chol = 240

#Using a one-sample t-test, this code calculates pval for alt hypothesis 'people with heart disease have an average 
# cholesterol level that is greater than 240 mg/dl'. I am using a significance threshold of 0.05.
tstat, pval = ttest_1samp(chol_hd, high_chol)
one_sided_pval = pval/2
print("P-value from 1-sample t-test: ", one_sided_pval)
# The p-value is small so we can reject the null hypothesis. The heart disease patients have a mean 
# cholesterol amount that exceeds the 240 by a statistically significant amount.

#pull chol levels from NO heart disease subset into variable and calc mean
no_chol_hd = no_hd['chol']
mean_no_chol_hd = np.mean(no_chol_hd)
print("The mean cholesterol levels of the patients without heart disease: ", mean_no_chol_hd)

#calculate pval for same alt hypothesis on chol data from no heart disease subset
tstat, pval2 = ttest_1samp(no_chol_hd, high_chol)
one_sided_pval2 = pval2/2
print("P-value from 1-sample t-test: ", one_sided_pval2)
#The p-value is large so we cannot reject the null hypothesis.The patients without heart disease have a mean 
# cholesterol amount that exceeds the 240 by a statistically insignificant amount.

#number of patients in heart data set
num_patients = len(heart)
print("Number of patients: ", num_patients)

#num patients with fasting bp > 120
num_highfbs_patients = np.sum(heart.fbs == 1)
print("Number of patients with high fasting blodd sugar: ", num_highfbs_patients)

#calc expected high fbs based on 8% US pop estimate
expected_fbs = 0.08*num_patients
print("Number of patients we'd expect to have high fbs: ", expected_fbs)


#By some estimates, about 8% of the U.S. population had diabetes. Fasting blood sugar levels greater than 120 mg/dl can be 
# indicative of diabetes. If this sample were representative of the population, approximately how many patients in the data
# should we expect to have diabetes?

#run binomial test with alt hypothesis: this sample (heart) was drawn from a population where more than
# 8% of people have fasting blood sugar > 120 mg/dl
p_value_1sided = binom_test(num_highfbs_patients, n=num_patients, p=.08, alternative='greater')
print("1-Sided p-value from binomial test: ", p_value_1sided)
#The p-value is very small, so we accept the alt hypothesis that this data was drawn from a population
# where more than 8 percent of people have high fasting blood sugar.


###In this second part of the project, I investigate the the association between 3 or more variables.

#Is thalach associated with whether or not a patient will ultimately be diagnosed with heart disease? 
# Side-by-side box plots give a general idea.
sns.boxplot(x=heart.heart_disease, y=heart.thalach)
plt.show()
plt.clf()

#comparing difference in means between thalach values for heart disease vs no heart disease
mean_differences = heart.groupby('heart_disease').mean()
print(mean_differences)

thalach_hd = heart.thalach[heart.heart_disease == 'Presence']
thalach_no_hd = heart.thalach[heart.heart_disease == 'Absence']

thal_mean_diff = np.mean(thalach_no_hd) - np.mean(thalach_hd)
thal_med_diff = np.median(thalach_no_hd) - np.median(thalach_hd)

print("difference in means of thalach for heart disease/no heart disease patients: " + str(round(thal_mean_diff, 2)))

print("difference in medians of thalach for heart disease/no heart disease patients: " + str(thal_med_diff))

#running two-sample t-test to determine if the average thalach of a heart disease patient is
#significantly different from the average thalach for a person without heart disease.
tstat, pval = ttest_ind(thalach_hd, thalach_no_hd)
print(pval)
#with a significant pval of 3.457e-14, we can reject to null hypothesis that states there is no
#significant difference between thalach for patients with/without heart disease.

#running through the same processes to investigate the relationship of the cholesterol and heart disease variables
sns.boxplot(x=heart.heart_disease, y=heart.chol)
plt.show()
plt.clf()

chol_hd = heart.chol[heart.heart_disease == 'Presence']
chol_no_hd = heart.chol[heart.heart_disease == 'Absence']

mean_chol_diff = np.mean(chol_hd) - np.mean(chol_no_hd)
med_chol_diff = np.median(chol_hd) - np.median(chol_no_hd)

print("difference in means of cholesterol levels for heart disease/no heart disease patients: " + str(round(mean_chol_diff, 2)))

print("difference in medians of cholesterol levels for heart disease/no heart disease patients: " + str(med_chol_diff))

tstat, pval = ttest_ind(chol_hd, chol_no_hd)
print(pval)

#with a pvalue of 0.139, we cannot reject the null hypothesis that there is no significant difference in cholesterol levels
# for heart disease/no heart disease patients. This is different than the 1-sample t-test I performed earlier which showed
# that heart disease patients had chol levels significantly higher than 240.

#Investigating thalach and cp. Side-by-side box plots of thalach for each chest pain type
sns.boxplot(x=heart.cp, y=heart.thalach)
plt.show()
plt.clf()

#creating variables holding patients with different chest pain types
thalach_typical = heart.thalach[heart.cp == 'typical angina']
thalach_asymptom = heart.thalach[heart.cp == 'asymptomatic']
thalach_nonangin = heart.thalach[heart.cp == 'non-anginal pain']
thalach_atypical = heart.thalach[heart.cp == 'atypical angina']

# Null hypothesis: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people all have the same average thalach.
# Alternative: People with typical angina, non-anginal pain, atypical angina, and asymptomatic people do not all have the same average thalach.
# I chose to use an ANOVA test since I have a non-binary categorical variable and a quantitative variable
fstat, pval = f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print(pval)
#with a pval of 1.907e-10, concluded that there is at least one pair of chest pain types (cp) 
#for which people with those pain types have significantly different thalach.

#which pair(s) of chest pain types have significantly different thalach? Below, I use the Tukey' range test:
tukey_results = pairwise_tukeyhsd(heart.thalach, heart.cp, 0.05)
print(tukey_results)
#the tukey results reveal that patients with "asymptomatic" chest pain type have significantly 
#different average thalach when compared with patients with any of the other three chest pain types

#investigating the relationship between the kind of chest pain a person experiences and whether or not they have heart disease.
#Using chi-squared analysis

#contingency table of cp and heart_disease
Xtab = pd.crosstab(heart.cp, heart.heart_disease)
print(Xtab)

chi2, pval, dof, expected = chi2_contingency(Xtab)
print(pval)
#with a pval of 1.25e-17, we can reject that null hypothesis that states there is no significant association 
#between chest pain type and whether or not someone is diagnosed with heart disease.