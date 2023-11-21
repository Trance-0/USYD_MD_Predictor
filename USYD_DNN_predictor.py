"""
This is just for fun I don't know if I build it or not.
I miss tensorflow so much.
"""
from USYD_data_preprocessing import USYD_data 

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

import numpy as np

seed = 39

data=USYD_data(debug=True)

# split training and validation
x_train, x_test, y_train, y_test = data.get_data(normalization=True,split=0.2)

# code referenced from: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/keras_tuner.ipynb#scrollTo=9E0BTp9Ealjb
def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(49,activation="relu",input_dim=7))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(5,activation='softmax'))

  # Tune the learning rate for the optimizer
  # Choose an optimal value
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=30,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print(np.array(x_train).shape,np.array(y_train).shape)

tuner.search(np.array(x_train), np.array(y_train), epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(np.array(x_train), np.array(y_train), epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)
pred_y=hypermodel.predict(data.pred_x).tolist()
print(pred_y)
print("prediction: ",data.Res_remapping[pred_y[0].index(1.0)])