# 绘制生存曲线
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def plot_lifeline(df, cluster_num, ax, title=''):
    kmf = KaplanMeierFitter()

    for i in range(cluster_num):
        idx = (df['Label'] == i)
        kmf.fit(df['Survival'][idx], df['Death'][idx], label='group' + str(i))
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
