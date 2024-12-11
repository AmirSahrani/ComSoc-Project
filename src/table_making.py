# %%

import pandas as pd

from plotting import load

# %%
dic = load("../results/stats_regression.pkl")
# I want the keys to not be used as the columns names, instead I want the first row (the voting rules to be the index) and the utility functions to be a column
df = pd.DataFrame(dic)
df.head()
# %%
piv = pd.pivot(df, index="voting_rule", columns="utility_function")
