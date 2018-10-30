import matplotlib.pyplot as plt
import numpy as np

log_path = "xxx"

with open(log_path, 'r') as f:
    log = list()
    idx = 0
    for line in f.readlines():
        if len(line) < 2 or line[0] != 's':
            continue
        _, loss, loc, conf = line.split('|')
        _, loss = loss.split(':')
        loss = float(loss)
        _, loc = loc.split(':')
        loc = float(loc)
        _, conf = conf.split(':')
        conf = float(conf)
        log.append([idx, loss, loc, conf])
        idx += 1

idx = np.array(log, dtype=np.int)[:, 0].tolist()
loss = np.array(log)[:, 1].tolist()
loc = np.array(log)[:, 2].tolist()
conf = np.array(log)[:, 3].tolist()

title = log_path.replace('\\', '/').split('/')[-1].split('.')[0]
title = 'log_loss ({})'.format(title)
plt.title(title)
plt.xlabel('epoch')
plt.plot(idx, loss, color='red', label='loss')
plt.plot(idx, loc, color='green', label='loc')
plt.plot(idx, conf, color='blue', label='conf')
plt.legend()
plt.grid()
plt.show()