class GlobalMean:
    def __init__(self):
        pass

    def train(self, df):
        self._mean = df['review/score'].mean()

    def predict(self, id):
        return self._mean


class PerBook4.601852Baseline:
    def __init__(self):
        self._seen = 0
        self._missing = 0

    def train(self, df):
        self._global_mean = df['review/score'].mean()
        self._means = df.groupby('Id').agg({'review/score': 'mean'})

    def predict(self, id):
        if id in self._means.index:
            self._seen += 1
            return self._means.loc[id, 'review/score']
        else:
            self._missing += 1
            return self._global_mean
