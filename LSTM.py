from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
from pandas import DataFrame
import json

df = pd.read_json('./data.json')
df.set_index('date',inplace=True)
fcst = df[-2:] #.values[0] => {today's day, sky, rain}, .values[1] => {tmr's day, sky, rain}
df = df[:-2]
today_set = np.concatenate([df[-1:].values[0],fcst.values[0][:-1]])
today_set = today_set.tolist()
today_set = [[today_set]]
tmr_set = np.concatenate([fcst.values[0],fcst.values[1][:-1]])
tmr_set= tmr_set.tolist()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

reframed = series_to_supervised(df, 1, 1) #t-1시점,t시점 데이터를 한 행으로 둔다

values = reframed.values
train = values[:50,:]
test = values[50:,:]

train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]


train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

model = Sequential()
model.add(LSTM(2000, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_x, train_y, epochs=20, batch_size=72, validation_data=(test_x, test_y), verbose=2, shuffle=False)
result = model.predict(today_set)
tmr_set[3]=int(result[0][0])
result2 = model.predict([[tmr_set]])
results = [str(result[0][0]),str(int(result2[0][0]))]
M = dict(zip(range(1, len(results) + 1), results))
jsonResult = json.dumps(M)
