from models import baseline
from data import data

def rmse(model, train, val):
    model.train(train)
    predictions = val['Id'].apply(lambda x: model.predict(x))
    errors = (predictions - val['review/score']) ** 2
#    import pdb;pdb.set_trace()
    return (errors.mean() ** 0.5)

if __name__ == '__main__':
    df = data.df('./data/Books_rating.csv')
    train, test, val = data.split_pd(df)
    print("Global RMSE", rmse(baseline.GlobalMean(), train, val))
    print("Per Book RMSE", rmse(baseline.PerBookBaseline(), train, val))
