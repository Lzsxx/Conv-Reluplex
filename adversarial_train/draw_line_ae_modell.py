
from skimage import io,transform
import tensorboard
import os
import glob
import numpy as np
import pylab
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
from brokenaxes import brokenaxes


# final
train_acc=[  0.9537,  0.9581,  0.9612,  0.9624,  0.9645,  0.9672,  0.9689,  0.97,  0.9713,  0.9723]
test_ae_acc = [ 0.9861111111111112,  0.9930555555555556,  0.9930555555555556,  0.9965277777777778,  0.9930555555555556,  0.9930555555555556,  0.9965277777777778,  0.9965277777777778,  0.9965277777777778,  0.9965277777777778]
test_original_acc = [ 0.9475,  0.9473,  0.9466,  0.9477,  0.9469,  0.9481,  0.9483,  0.948,  0.9485,  0.9489]



# fig,ax = plt.subplots()
# ax.set_xlim([1,11])
# ax.set_ylim(9,10)

x1 = range(1, 11)

y1 = train_acc
y2 = test_ae_acc
y3 = test_original_acc

font1 = {
'weight' : 'normal',
'size'   : 16,
}

fig = plt.figure(figsize=(8, 6))
plt.tick_params(labelsize=14)
plt.plot(x1, y1, 'o-',label="train accuracy")
plt.plot(x1, y2, 's-', label="adversarial example test accuracy ", color='orange')
plt.plot(x1, y3, 'v-', label="original example test accuracy ", color='green')
plt.title('Accuracy comparison', font1)
plt.xlabel('Epoches', font1)
plt.ylabel('accuracy', font1)
plt.legend()
plt.savefig("./Accuracy-comparison.jpg")
plt.show()




















# # ax0 = plt.subplot(gs[0])
# # ax0.plot(x, y)
# # ax1 = plt.subplot(gs[1])
# # ax1.plot(y, x)
# # plt.tight_layout()
#
# # fig = plt.figure(figsize=(8, 6))
# gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
# ax = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# # plt.tight_layout()
#
# # use the top (ax) for the outliers, and the bottom
# # (ax2) for the details of the majority of our data
# # f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
#
# # plot the same data on both axes
# ax.plot(test_ae_acc)
# ax2.plot(test_ae_acc)
#
# # zoom-in / limit the view to different portions of the data
# ax.set_ylim(0.92, 1.01)  # outliers only
# ax2.set_ylim(0, 0.01)  # most of the data
#
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
# ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
#
# # hide the spines between ax and ax2
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()
#
#
# d = 0.01  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
# # What's cool about this is that now if we vary the distance between
# # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# # the diagonal lines will move accordingly, and stay right at the tips
# # of the spines they are 'breaking'
# plt.show()