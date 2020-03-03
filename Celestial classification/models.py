import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from data_processing import DataProcessing
random_seed = 0

train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"
sample_submission_data_path = "./data/sample_submission.csv"

data_processing = DataProcessing(train_data_path, test_data_path, sample_submission_data_path)
train_data, test_data, sample_submission_data = data_processing.load_file()
x_train, x_valid, y_train, y_valid = data_processing.set_data(train_data, test_data)

'''
# catboost
cat_clf = CatBoostClassifier(iterations = 20000, random_state = random_seed, task_type="GPU")
cat_clf.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_valid, y_valid)])
cat_pred = cat_clf.predict_proba(test_data)
submission = pd.DataFrame(data=cat_pred, columns=sample_submission_data.columns, index=sample_submission_data.index)
submission.to_csv('./results/cat_boost2.csv', index=True)
'''

# lgbm
#lgbm_clf = LGBMClassifier(n_estimators = 1000, n_jobs=-1, random_state = random_seed, device = 'gpu')
lgbm_clf = LGBMClassifier(n_estimators = 1000, n_jobs=-1, random_state = random_seed)
lgbm_clf.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_valid, y_valid)])
lgbm_pred = lgbm_clf.predict_proba(test_data)
submission = pd.DataFrame(data=lgbm_pred, columns=sample_submission_data.columns, index=sample_submission_data.index)
submission.to_csv('./results/light_gbm2.csv', index=True)

# xgboost
#xgb_clf = XGBClassifier(n_estimators = 1000, n_jobs=-1, random_state=random_seed, tree_method='gpu_exact')
xgb_clf = XGBClassifier(n_estimators = 1000, n_jobs=-1, random_state=random_seed)
xgb_clf.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_valid, y_valid)])
xgb_pred = xgb_clf.predict_proba(test_data)
submission = pd.DataFrame(data=xgb_pred, columns=sample_submission_data.columns, index=sample_submission_data.index)
submission.to_csv('./results/xg_boost2.csv', index=True)
