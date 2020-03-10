import pandas as pd

from training_plot import plots

scores = pd.read_csv("chemprop-master/data/qm7.csv", usecols=["u0_atom"])
preds = pd.read_csv("chemprop-master/data/qm7_preds.csv", usecols=["u0_atom"])

plots(scores, preds, "QM7")
