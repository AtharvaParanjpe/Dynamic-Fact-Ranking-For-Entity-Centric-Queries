import pandas as pd
from heatmap import heatmap
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
sns.set(color_codes=True, font_scale=1.2)

df = pd.read_csv("Complete_Data.csv")
df = df.replace({'rel': {0: 'Not-relevant',1:'Relevant',2:'Very-Relevant'}})
df = df.replace({'imp': {0: 'Not-important',1:'Important',2:'Very-Important'}})
df['cnt'] = np.ones(len(df))


g = df.groupby(['rel', 'imp']).count()[['cnt']].replace(np.nan, 0).reset_index()
print(g)
plt.rcParams["figure.figsize"] = (5,5)
heatmap(
    x=g['rel'], # Column to use as horizontal dimension 
    y=g['imp'], # Column to use as vertical dimension
    size_scale=7900, # Change this to see how it affects the plot
    # x_order=['0', '1', '2'], # Sort order for x labels
    # y_order=['Not important','Important','Very Important'], # Sort order for y labels
    color=g['cnt'], # Values to map to color, here we use number of items in each bucket
    palette=sns.cubehelix_palette(125) # We'll use black->red palette
)
# input()
plt.show()