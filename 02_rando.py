import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame creation (replace this with your actual DataFrame)
data = {
    'PROV_BIL_NPI': ['1234', '5678', '1234', '5678', '91011', '91011'],
    'year': [2020, 2020, 2020, 2020, 2020, 2020],
    'month': [1, 1, 2, 2, 1, 2],
    'percentage_90837': [65, 70, 60, 80, 55, 75]
}
df = pd.DataFrame(data)

# Create a 'month_year' column for plotting
df['month_year'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str)).dt.to_period('M')

# Plotting
g = sns.FacetGrid(df, col='month_year', col_wrap=4, height=4)
g.map(plt.hist, 'percentage_90837', bins=10, color='blue', alpha=0.7)
g.set_titles('{col_name}')
g.set_axis_labels('Percentage of 90837 Claims', 'Frequency')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Monthly Distribution of CPT_PROC_CD "90837" Claim Percentages')
plt.show()
