from  sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

datafile = u'F:\\N6-methyladenosine sites\\m6A\\Feature extraction\\Finally\\Saccgaromyces cerevisiae.csv'
dataset = pd.read_csv(datafile,header = None)
X = dataset.values[:, 0:530]
y = dataset.values[:, 531]

GBDT = GradientBoostingClassifier()
print(GBDT.fit(X,y))
print(GBDT.feature_importances_)
new_data1 = X[:, GBDT.feature_importances_>0]
print(new_data1.shape)
pd.DataFrame(new_data1).to_csv('F:\\N6-methyladenosine sites\\m6A\\Fusion\\GBDT\\Saccgaromyces cerevisiae.csv', header = False, index = False)
