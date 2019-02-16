import numpy as np
import pylab as plt



mean_mccs = [0.316, 0.264, 0.277, 0.240, 0.205]
max_mccs = [0.332, 0.276, 0.292, 0.276, 0.22996]
stds = [0.012, 0.005, 0.008, 0.016, 0.016]

mean_mccs.reverse()
max_mccs.reverse()
stds.reverse()

# accs = [71.68391, 70.83725, 69.52023, 70.74318, 69.7]



fig, ax = plt.subplots()
ind = np.arange(len(max_mccs))

x = [100, 300, 1000, 3000, 8551]
# y =

p1 = ax.semilogx(x, max_mccs, color='m')
p2 = ax.errorbar(x, mean_mccs, yerr=stds, color='c', ecolor='c', capsize=2)
ax.set_ylabel('Overall Dev MCC', fontsize=16)
plt.xlabel('Training Set Size', fontsize=16)

ax.set_xticks(x)
ax.set_xticklabels([100, 300, 1000, 3000, 8551])



for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)

ax.legend((p1[0], p2[0]), ('Max', 'Mean'), loc=2, fontsize=14)


# ax2 = ax.twinx()
# p1 = ax2.plot(ind, accs, color='y')


plt.show()





#
# mean_mccs = [0.316, 0.264, 0.277, 0.240, 0.205]
# max_mccs = [0.332, 0.276, 0.292, 0.276, 0.22996]
# stds = [0.012, 0.005, 0.008, 0.016, 0.016]
#
# mean_mccs.reverse()
# max_mccs.reverse()
# stds.reverse()
#
# # accs = [71.68391, 70.83725, 69.52023, 70.74318, 69.7]
#
#
#
# fig, ax = plt.subplots()
# ind = np.arange(len(max_mccs))
#
# x = [100, 300, 1000, 3000, 8551]
# # y =
#
# ax.set_xticks(ind)
# ax.set_xticklabels([100, 300, 1000, 3000, 8551])
# p1 = ax.plot(ind, max_mccs, color='m')
# p2 = ax.errorbar(ind, mean_mccs, yerr=stds, color='c', ecolor='c', capsize=2)
# ax.set_ylabel('Overall Dev MCC', fontsize=16)
# plt.xlabel('Training Set Size', fontsize=16)
#
#
#
#
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontsize(14)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontsize(14)
#
# ax.legend((p1[0], p2[0]), ('Max', 'Mean'), loc=2, fontsize=14)
#
#
# # ax2 = ax.twinx()
# # p1 = ax2.plot(ind, accs, color='y')
#
#
# plt.show()
