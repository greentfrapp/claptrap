from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from BinaryData import BinaryData

COLOR_EMERALD = "#1abc9c"
COLOR_ORANGE = "#f39c12"

def main():

	fig, ax = plt.subplots()
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	label_types = np.array([0, 1])
	width = 0.1
	color = {"variable": COLOR_ORANGE, "fixed": COLOR_EMERALD}

	size = 100000
	max_seq_len = 50

	for i, length_type in enumerate(["variable", "fixed"]):
		dataset = BinaryData(length_type=length_type, size=size, max_seq_len=max_seq_len)
		_, labels = dataset.next(dataset.size)
		ax.bar(label_types + i * width, [float(np.sum(labels == 0))/size, float(np.sum(labels == 1))/size], width, color=color[length_type], edgecolor=color[length_type], label=length_type)

	ax.set_xticks(label_types + width)
	ax.set_xticklabels(label_types)
	ax.set_xlim([-0.3,1.5])
	ax.set_xlabel('Label')
	ax.set_ylabel('Distribution')
	ax.legend(loc='upper left', frameon=False)

	fig.savefig('images/label_dist.png', bbox_inches='tight', dpi=300)

if __name__ == "__main__":
	main()