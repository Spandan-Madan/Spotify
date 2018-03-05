from collections import OrderedDict
import numpy as np
from .features import *
from .models import *
from .data.metrics import *
# initial settings
features = []
model = ''
plist = ''
# vectorize data
x_data = []
for feature in features:
    feature.load()
    x_data.append(feature.transform(plist))
X = np.hstack(x_data)
# model and recommend
model.load()
rec = model.recommend(X, n_total=x)

# define eval metrics
metrics = OrderedDict()

# compute stats
stats = OrderedDict()
for key, metric in metrics.items():
    stats[key] = metric(plist, rec)
# print results


