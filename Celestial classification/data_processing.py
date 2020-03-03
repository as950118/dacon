import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
random_seed = 0

train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"
sample_submission_data_path = "./data/sample_submission.csv"


# load file
class DataProcessing:
    def __init__(self, train_data_path, test_data_path, sample_submission_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.sample_submission_data_path = sample_submission_data_path

    def load_file(self):
        train_data = pd.read_csv(self.train_data_path, index_col=0)
        test_data = pd.read_csv(self.test_data_path, index_col=0)
        sample_submission_data = pd.read_csv(self.sample_submission_data_path, index_col=0)

        col_num = {col:i for i, col in enumerate(sample_submission_data.columns)}
        to_num = lambda x, dic: dic[x]
        train_data['type_num'] = train_data['type'].apply(lambda x: to_num(x, col_num))
        return train_data, test_data, sample_submission_data

    # set data
    def set_data(self, train_data, test_data):
        train_x = train_data.drop(columns = ['type', 'type_num'], axis=1)
        train_y = train_data['type_num']
        test_x = test_data
        x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1, random_state = random_seed, stratify = train_y)
        return x_train, x_valid, y_train, y_valid