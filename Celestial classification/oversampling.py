from imblearn.over_sampling import ADASYN
import numpy as np
import csv
import pandas as pd
reader = lambda rfname: list(csv.reader(open(rfname), delimiter=','))
writer = lambda wfname: csv.writer(open(wfname, 'w', newline=''))


class Data:
    def __init__(self, rfname, x_col, y_col, len_data=900000):
        data = reader(rfname)
        self.len_data = min(len_data, len(data))
        self.x_col_start = x_col[0]
        self.x_col_end = x_col[1]
        self.y_col = y_col

        # feature(x_data)와 result(y_data) 나누기
        self.col = [elem for elem in data[0][3:]] + ['type']
        self.x_data = np.array([list(map(float, [e for e in elem[self.x_col_start:self.x_col_end]])) for elem in data[1:self.len_data]])
        self.y_data = np.array([str(elem[self.y_col]) for elem in data[1:self.len_data]])

    # 오버샘플링
    def oversampling(self):
        over_x_data, over_y_data = ADASYN().fit_resample(self.x_data, self.y_data)
        over_data = np.concatenate((over_x_data, over_y_data[:, np.newaxis]), axis=1)
        dataframe = pd.DataFrame(data=over_data.tolist(), columns=self.col)
        dataframe.to_csv('oversampling.csv', encoding='euc-kr', index=0)

        #dataframe = pd.DataFrame(data=over_x_data.tolist(), columns=self.col)
        #dataframe = pd.DataFrame(data=over_x_data.tolist())
        #dataframe.to_csv('oversampling_x.csv', encoding='euc-kr', index=0)
        #dataframe = pd.DataFrame(data=over_y_data.tolist())
        #dataframe.to_csv('oversampling_y.csv', encoding='euc-kr', index=0)
        #np.savetxt('oversampling.csv', np.concatenate((over_x_data, over_y_data[:, np.newaxis]), axis=1), delimiter=",")



if __name__ == "__main__":
    Data(rfname="./data/train.csv", x_col = [3, 23], y_col = 1).oversampling()
