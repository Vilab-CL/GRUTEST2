import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


path = 'out.txt'

target_names = ['Temp']
na_values = ['999', '999.0', '999.9', '9999.9']

df_raw = pd.read_csv(path, sep=',', header=0,index_col=False, na_values=na_values)
df = pd.DataFrame()

df['Temp'] = df_raw['Temp']


shift_days = 1
shift_steps = shift_days * 1 # Number of hours.

df_targets = df.shift(-shift_steps)
df.head(shift_steps + 5)
x_data = df.values[0:-shift_steps]



print(x_data[0])

# In[29]:


print(type(x_data))
print("Shape:", x_data.shape)


# These are the output-signals (or target-signals):

# In[30]:


y_data = df_targets.values[:-shift_steps]


# In[31]:


print(type(y_data))
print("Shape:", y_data.shape)


# This is the number of observations (aka. data-points or samples) in the data-set:

# In[32]:


num_data = len(x_data)
num_data


# This is the fraction of the data-set that will be used for the training-set:

# In[33]:


train_split = 0.75




num_train = int(train_split * num_data)
num_train


# This is the number of observations in the test-set:

# In[35]:


num_test = num_data - num_train
num_test


# These are the input-signals for the training- and test-sets:

# In[36]:


x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)


# These are the output-signals for the training- and test-sets:

# In[37]:


y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)


# This is the number of input-signals:

# In[38]:


num_x_signals = x_data.shape[1]
num_x_signals


# This is the number of output-signals:

# In[39]:


num_y_signals = y_data.shape[1]
num_y_signals


# ### Scaled Data
# 
# The data-set contains a wide range of values:

# In[40]:


print("Min:", np.min(x_train))
print("Max:", np.max(x_train))


print(df.head(5))
print(df_targets.tail())

# df['Temp'].plot()
# plt.show()

# input()


# exit()


# The neural network works best on values roughly between -1 and 1, so we need to scale the data before it is being input to the neural network. We can use `scikit-learn` for this.
# 
# We first create a scaler-object for the input-signals.

# In[41]:


x_scaler = MinMaxScaler()


# We then detect the range of values from the training-data and scale the training-data.

# In[42]:


x_train_scaled = x_scaler.fit_transform(x_train)


# Apart from a small rounding-error, the data has been scaled to be between 0 and 1.

# In[43]:


print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))



# We use the same scaler-object for the input-signals in the test-set.

# In[44]:


x_test_scaled = x_scaler.transform(x_test)


# The target-data comes from the same data-set as the input-signals, because it is the weather-data for one of the cities that is merely time-shifted. But the target-data could be from a different source with different value-ranges, so we create a separate scaler-object for the target-data.

# In[45]:


y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ## Data Generator
# 
# The data-set has now been prepared as 2-dimensional numpy arrays. The training-data has almost 300k observations, consisting of 20 input-signals and 3 output-signals.
# 
# These are the array-shapes of the input and output data:

# In[46]:


print(x_train_scaled.shape)
print(y_train_scaled.shape)




def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)


# We will use a large batch-size so as to keep the GPU near 100% work-load. You may have to adjust this number depending on your GPU, its RAM and your choice of `sequence_length` below.

# In[48]:


batch_size = 128


# We will use a sequence-length of 1344, which means that each random sequence contains observations for 8 weeks. One time-step corresponds to one hour, so 24 x 7 time-steps corresponds to a week, and 24 x 7 x 8 corresponds to 8 weeks.

# In[49]:


sequence_length = 3
sequence_length


# We then create the batch-generator.

# In[50]:


generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)


# We can then test the batch-generator to see if it works.

# In[51]:


x_batch, y_batch = next(generator)



print(x_batch.shape)
print(y_batch.shape)



# We can plot one of the 20 input-signals as an example.

# In[53]:


batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
#plt.plot(seq)


# We can also plot one of the output-signals that we want the model to learn how to predict given all those 20 input signals.

# In[54]:


seq = y_batch[batch, :, signal]
#plt.plot(seq)



# ### Validation Set
# 
# The neural network trains quickly so we can easily run many training epochs. But then there is a risk of overfitting the model to the training-set so it does not generalize well to unseen data. We will therefore monitor the model's performance on the test-set after each epoch and only save the model's weights if the performance is improved on the test-set.
# 
# The batch-generator randomly selects a batch of short sequences from the training-data and uses that during training. But for the validation-data we will instead run through the entire sequence from the test-set and measure the prediction accuracy on that entire sequence.

# In[55]:


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


#plt.show()

model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))



model.add(Dense(num_y_signals, activation='sigmoid'))


if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))




warmup_steps = 50


# In[61]:


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


# ### Compile Model
# 
# This is the optimizer and the beginning learning-rate that we will use.

# In[62]:


optimizer = RMSprop(lr=1e-3)


# We then compile the Keras model so it is ready for training.

# In[63]:


model.compile(loss=loss_mse_warmup, optimizer=optimizer)


# This is a very small model with only two layers. The output shape of `(None, None, 3)` means that the model will output a batch with an arbitrary number of sequences, each of which has an arbitrary number of observations, and each observation has 3 signals. This corresponds to the 3 target signals we want to predict.

# In[64]:


model.summary()


# ### Callback Functions
# 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# 
# This is the callback for writing checkpoints during training.

# In[65]:


path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_weights_only=True,
                                       save_best_only=True)


# This is the callback for stopping the optimization when performance worsens on the validation-set.

# In[66]:


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)


# This is the callback for writing the TensorBoard log during training.

# In[67]:


callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)


# This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since the last epoch (as indicated by `patience=0`). The learning-rate will be reduced by multiplying it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.

# In[68]:


callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)


# In[69]:


callbacks = [#callback_early_stopping,
            callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# ## Train the Recurrent Neural Network
# 
# We can now train the neural network.
# 
# Note that a single "epoch" does not correspond to a single processing of the training-set, because of how the batch-generator randomly selects sub-sequences from the training-set. Instead we have selected `steps_per_epoch` so that one "epoch" is processed in a few minutes.
# 
# With these settings, each "epoch" took about 2.5 minutes to process on a GTX 1070. After 14 "epochs" the optimization was stopped because the validation-loss had not decreased for 5 "epochs". This optimization took about 35 minutes to finish.
# 
# Also note that the loss sometimes becomes `NaN` (not-a-number). This is often resolved by restarting and running the Notebook again. But it may also be caused by your neural network architecture, learning-rate, batch-size, sequence-length, etc. in which case you may have to modify those settings.

# In[ ]:


#get_ipython().run_cell_magic('time', '', 'model.fit_generator(generator=generator,\n                    epochs=20,\n                    steps_per_epoch=100,\n                    validation_data=validation_data,\n                    callbacks=callbacks)')


model.fit_generator(generator=generator,
                    epochs=8,
                    steps_per_epoch=500,
                    validation_data=validation_data,
                    callbacks=callbacks)

#exit()


# ### Load Checkpoint
# 
# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.

# In[70]:


try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# ## Performance on Test-Set
# 
# We can now evaluate the model's performance on the test-set. This function expects a batch of data, but we will just use one long time-series for the test-set, so we just expand the array-dimensionality to create a batch with that one sequence.

# In[71]:


result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


# In[72]:


print("loss (test-set):", result)


# In[73]:


# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))


# ## Generate Predictions
# 
# This helper-function plots the predicted and true output-signals.

# In[74]:


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    print (y_pred_rescaled)
    print (y_true)

    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# We can now plot an example of predicted output-signals. It is important to understand what these plots show, as they are actually a bit more complicated than you might think.
# 
# These plots only show the output-signals and not the 20 input-signals used to predict the output-signals. The time-shift between the input-signals and the output-signals is held fixed in these plots. The model **always** predicts the output-signals e.g. 24 hours into the future (as defined in the `shift_steps` variable above). So the plot's x-axis merely shows how many time-steps of the input-signals have been seen by the predictive model so far.
# 
# The prediction is not very accurate for the first 30-50 time-steps because the model has seen very little input-data at this point.
# The model generates a single time-step of output data for each time-step of the input-data, so when the model has only run for a few time-steps, it knows very little of the history of the input-signals and cannot make an accurate prediction. The model needs to "warm up" by processing perhaps 30-50 time-steps before its predicted output-signals can be used.
# 
# That is why we ignore this "warmup-period" of 50 time-steps when calculating the mean-squared-error in the loss-function. The "warmup-period" is shown as a grey box in these plots.
# 
# Let us start with an example from the training-data. This is data that the model has seen during training so it should perform reasonably well on this data.

# In[75]:


plt.show()
plot_comparison(start_idx=1, length=2318, train=True)



# The model was able to predict the overall oscillations of the temperature quite well but the peaks were sometimes inaccurate. For the wind-speed, the overall oscillations are predicted reasonably well but the peaks are quite inaccurate. For the atmospheric pressure, the overall curve-shape has been predicted although there seems to be a slight lag and the predicted curve has a lot of noise compared to the smoothness of the original signal.

# ### Strange Example
# 
# The following is another example from the training-set.
# 
# Note how the temperature does not oscillate very much within each day (this plot shows almost 42 days). The temperature normally oscillates within each day, see e.g. the plot above where the daily temperature-oscillation is very clear. It is unclear whether this period had unusually stable temperature, or if perhaps there's a data-error.

# In[76]:


#plot_comparison(start_idx=200000, length=1000, train=True)


# As a check, we can plot this signal directly from the resampled data-set, which looks similar.

# In[77]:





# ### Example from Test-Set
# 
# Now consider an example from the test-set. The model has not seen this data during training.
# 
# The temperature is predicted reasonably well, although the peaks are sometimes inaccurate.
# 
# The wind-speed has not been predicted so well. The daily oscillation-frequency seems to match, but the center-level and the peaks are quite inaccurate. A guess would be that the wind-speed is difficult to predict from the given input data, so the model has merely learnt to output sinusoidal oscillations in the daily frequency and approximately at the right center-level.
# 
# The atmospheric pressure is predicted reasonably well, except for a lag and a more noisy signal than the true time-series.

# In[79]:


#plot_comparison(start_idx=200, length=1000, train=False)


input()
