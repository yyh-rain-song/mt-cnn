from keras.layers import Conv2DTranspose, Conv2D, Dropout, MaxPooling2D, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from load_data import load_data
import math

X, Y = load_data()
print(X.shape, Y.shape)
x_train = X[:480]
x_test = X[480:]
y_train = Y[:480]
y_test = Y[480:]


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:], data_format='channels_last'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2DTranspose(32, (3, 3), padding='same', data_format='channels_last'))
model.add(Dense(units=1))
opt = Adam()
model.compile(optimizer=opt, loss=mean_squared_error)
# model.summary()
model.fit(x_train, y_train, epochs=50)

score = model.evaluate(x_test, y_test)
print(math.sqrt(score))