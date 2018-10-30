import math


"""
  Getting the min_sizes and max sizes of default boxes 
"""
min_sizes = []
max_sizes = []
min_dim = 300
mbox_source_layers = 6
# # in percent %
min_ratio = 20
max_ratio = 90
step = int(math.floor((max_ratio - min_ratio) / (mbox_source_layers - 2)))
for ratio in range(min_ratio, max_ratio+1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
max_sizes = [min_dim * 20 / 100.] + max_sizes

# 416 13 26 52
# 320 40 20 10

print(min_sizes)
print(max_sizes)
