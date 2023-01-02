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

df = read_csv('ex1.csv', sep=';')
# df.plot(x='Input Length in Multiples of 131070', kind="bar")
xl = "Vector length"
cl = "Three segments"
bl = "Two segments"
al = "One segment"
x = df[xl]
c = df[cl]
b = df[bl]
a = df[al]
w = 1.6
fig, ax = plt.subplots()
to = ax.plot((x), a, label=al)
ke = ax.plot((x), b, label=bl)
fr = ax.plot((x), c, label=cl)
plt.xlabel(xl)
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
