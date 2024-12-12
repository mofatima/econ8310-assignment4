import pandas as pd
import pymc as pm
import arviz as az

url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv"
data = pd.read_csv(url)

print("Dataset Preview:")
print(data.head())

grouped_data = data.groupby('version').agg(
    retention1_mean=('retention_1', 'mean'),
    retention1_count=('retention_1', 'count'),
    retention7_mean=('retention_7', 'mean'),
    retention7_count=('retention_7', 'count')
).reset_index()

print("\nGrouped Data Summary:")
print(grouped_data)

print("\nStarting Bayesian analysis for 1-day retention...")
with pm.Model() as retention1_model:
    p_control = pm.Beta("p_control", alpha=1, beta=1)
    p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)
    control_data = data[data['version'] == 'gate_30']['retention_1']
    treatment_data = data[data['version'] == 'gate_40']['retention_1']
    obs_control = pm.Binomial("obs_control", n=len(control_data), p=p_control, observed=control_data.sum())
    obs_treatment = pm.Binomial("obs_treatment", n=len(treatment_data), p=p_treatment, observed=treatment_data.sum())
    diff_retention1 = pm.Deterministic("diff_retention_1", p_treatment - p_control)
    retention1_trace = pm.sample(2000, return_inferencedata=True)

print("\nStarting Bayesian analysis for 7-day retention...")
with pm.Model() as retention7_model:
    p_control = pm.Beta("p_control", alpha=1, beta=1)
    p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)
    control_data = data[data['version'] == 'gate_30']['retention_7']
    treatment_data = data[data['version'] == 'gate_40']['retention_7']
    obs_control = pm.Binomial("obs_control", n=len(control_data), p=p_control, observed=control_data.sum())
    obs_treatment = pm.Binomial("obs_treatment", n=len(treatment_data), p=p_treatment, observed=treatment_data.sum())
    diff_retention7 = pm.Deterministic("diff_retention_7", p_treatment - p_control)
    retention7_trace = pm.sample(2000, return_inferencedata=True)

print("\nSummarizing results for 1-day retention:")
summary_retention1 = az.summary(retention1_trace, var_names=["diff_retention_1"])
print(summary_retention1)

print("\nSummarizing results for 7-day retention:")
summary_retention7 = az.summary(retention7_trace, var_names=["diff_retention_7"])
print(summary_retention7)

print("\nVisualizing posterior distributions...")
az.plot_posterior(retention1_trace, var_names=["diff_retention_1"], hdi_prob=0.95)
az.plot_posterior(retention7_trace, var_names=["diff_retention_7"], hdi_prob=0.95)
