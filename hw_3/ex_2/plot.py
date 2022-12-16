#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import numpy as np
import scipy.stats as st
from pandas import read_csv

df = read_csv('double.csv', sep=';')
# df.plot(x='Input Length in Multiples of 131070', kind="bar")
x = df['Input values(RowA, ColA=RowB, ColB)']
c = df['To Device']
b = df['Kernel']
a = df['From Device']
w = 0.8
fig, ax = plt.subplots()
to = ax.bar((x), a, width=w, label='To Device')
ke = ax.bar((x), b, bottom=a, width=w, label='Kernel')
fr = ax.bar((x), c, bottom=a+b, width=w, label='From Device')
plt.xlabel("Input values(RowA, ColA=RowB, ColB)")
plt.ylabel("Time in microseconds")
ax.legend()
# ax.set_xscale("log")
plt.show()
# d = [list(row) for row in df.values]

# x = range(length+1)[1:]
# y = np.array(d2)

# xlog = np.log10(x)
# ylog = np.log10(y)
# k, m = np.polyfit(xlog, ylog, 1)
# linreg = np.poly1d((m,k))(x)
# linreg = list(map(lambda v : 10.0 ** m * v ** k, x))
# print("Plotting")

# fig, ax = plt.subplots()
# # ax.plot(d[:,0], d[:,1])
# ax.set_xlabel("Word Rank")
# ax.set_ylabel("Word Frequency")
# # ax.legend(["Data", "Fit - k: " + format(k, '.3f') + " m: " + format(m, '.3f')])
# # plt.xticks(range(length), t11, size='small')
# plt.show()
# plt.savefig('histogram.pgf')

import code
code.interact(local=dict(globals(), **locals()))
