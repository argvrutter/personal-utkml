import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

# From here: https://github.com/Dataweekends/zero_to_deep_learning_udemy/blob/master/data/weight-height.csv
train = pd.read_csv('train1.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
train.plot(kind='scatter', x='a bunch of stuff', y='historical_popularity_index', title='Popularity of Historical Figures')
x_train = train[['average_views'], ['article_languages'], ['birth_year'], ['page_views'], ['average_views']].values.astype(np.float32)
y_train = train['historical_popularity_index'].values.astype(np.float32)
x_train.fillna(0)
# dtypes.columns -> give everything
# .dtypes data types of columns
print train.dtypes
y_train = np.reshape(y_train, (-1, len(x_train)))
x_train = np.reshape(y_train, (-1, 1 ))
x_train['birth_years'].apply(pd.to_numeric, errors='coerce')
print(x_train.shape)
# Hyper Parameters
print(y_train.shape)
input_size = 5
output_size = 1
num_epochs = 600
learning_rate = 0.001

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size, output_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#.apply(pd.to_numeric, errors='coerce') -> string to numerical
# Train the Model
for epoch in range(num_epochs):
    # Convert numpy array to torch Variable
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print ('Epoch [%d/%d], Loss: %.4f'
               %(epoch+1, num_epochs, loss.data[0]))


# Plot the graph
y_train = np.reshape(y_train, (-1, 1 ))
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'bo', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

test['historical_popularity_index'] = predicted
test[['historical_popularity_index']].to_csv('actual_submission.csv')
test = tst
