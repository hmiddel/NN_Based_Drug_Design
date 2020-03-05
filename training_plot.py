import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("data/predictions.tsv", sep="\t")
data=data.dropna()
x=np.log10(pd.to_numeric(data["IC50 (nM)"]))
y=np.log10(pd.to_numeric(data["predicted IC50"]))
fig = plt.figure()
#axes= fig.add_axes([0.1,0.1,0.8,0.8])
#fig = plt.scatter(x,y)
#axes.set_xlim([-2,5])
#axes.set_ylim([-2,5])
#plt.show()


plt.hist(y)
plt.show()



