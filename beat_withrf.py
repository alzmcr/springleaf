import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing, cross_validation
from sklearn.metrics import roc_auc_score as auc
from time import time

# PREPARE DATA
data = pd.read_csv('data/train.csv').set_index("ID")
test = pd.read_csv('data/test.csv').set_index("ID")

# remove constants
nunique = pd.Series([data[col].nunique() for col in data.columns], index = data.columns)
constants = nunique[nunique<2].index.tolist()
data = data.drop(constants,axis=1)
test = test.drop(constants,axis=1)

# encode string
strings = data.dtypes == 'object'; strings = strings[strings].index.tolist(); encoders = {}
for col in strings:
    encoders[col] = preprocessing.LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])
    try:
        test[col] = encoders[col].transform(test[col])
    except:
        # lazy way to incorporate the feature only if can be encoded in the test set
        del test[col]
        del data[col]

# DATA ready
X = data.drop('target',1).fillna(0); y = data.target

# RF FTW :)
rf = ensemble.RandomForestClassifier(n_jobs=4, n_estimators = 20, random_state = 11)

# CROSS VALIDATE AND PRINT TRAIN AND TEST SCORE
kf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=11)
trscores, cvscores, times = [], [], []
for itr, icv in kf:
    t = time()
    trscore = auc(y.iloc[itr], rf.fit(X.iloc[itr], y.iloc[itr]).predict_proba(X.iloc[itr])[:,1])
    cvscore = auc(y.iloc[icv], rf.predict_proba(X.iloc[icv])[:,1])
    trscores.append(trscore); cvscores.append(cvscore); times.append(time()-t)
print "TRAIN %.4f | TEST %.4f | TIME %.2fm (1-fold)" % (np.mean(trscores), np.mean(cvscores), np.mean(times)/60)

# MAKING SUBMISSION
submission = pd.DataFrame(rf.fit(X,y).predict_proba(test.fillna(0))[:,1], index=test.index, columns=['target'])
submission.index.name = 'ID'
submission.to_csv('beat_withrf.csv')


