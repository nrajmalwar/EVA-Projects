```python
tf.enable_eager_execution()
```
Let's first look at what happens when we don't use eager mode. All the numbers, tensors and operations are part of the computational graph. Any variable is evaluated only when the entire graph is evaluated. When the time comes to debug your code, this is not practical as you cannot evaluate and check each line. It is always useful to be able to print out partial code or iterate over few lines of code and get more meaningful outputs.

In eager mode, tensorflow performs operations eagerly, that is, it is evaluated right there rather than waiting for the entire operations to complete.

---
```python
BATCH_SIZE = 512
MOMENTUM = 0.9
LEARNING_RATE = 0.4
WEIGHT_DECAY = 5e-4
EPOCHS = 24
```
These are the hyperparameters we use in our model

---

```python
def init_pytorch(shape, dtype=tf.float32, partition_info=None):
  # shape = shape of the variable to initialize
  # dtype = dtype of generated values
  
  fan = np.prod(shape[:-1])
  bound = 1 / math.sqrt(fan)
  return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)
```
We need to initialize the weights exactly the way it is done in DavidNet. Different initialization lead to different scales, which means we might need different set of hyperparameters for different weight initialization. Keras by default uses Xavier Glorot initialization whereas DavidNet uses plain initialization. Since DavidNet is written in Pytorch, we use the same initialization function, implemented for Keras.

---
```python
class ConvBN(tf.keras.Model):
  def __init__(self, c_out):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)-
    self.drop = tf.keras.layers.Dropout(0.05)

  def call(self, inputs):
    return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))
 ```
 We define a Convolution Block in tf keras using a class. We define two functions, "init" function to create layer objects and "call" function to expresses the logic of the class.

We use the default keras BatchNorm which is exactly the same as what DavidNet has created in PyTorch. The momentum and epsilon terms are matched with DavidNet.

ReLu is not a layer since it does not store weights, it is rather an operation performed on a layer.

---
```python
class ResBlk(tf.keras.Model):
  def __init__(self, c_out, pool, res = False):
    super().__init__()
    self.conv_bn = ConvBN(c_out)
    self.pool = pool
    self.res = res
    if self.res:
      self.res1 = ConvBN(c_out)
      self.res2 = ConvBN(c_out)

  def call(self, inputs):
    h = self.pool(self.conv_bn(inputs))
    if self.res:
      h = h + self.res2(self.res1(h))
    return h
 ```
 The residual block in DavidNet is written as a class. In the "init" function, we call the ConvBN function with the required number of channels (c_out) and check whether a pool layer and a residual layer (with skip connection) is required. If residual layer is true, we use two ConvBN layer as defined in the model.

In the "call" function, we create the logic for the residual block layer with the skip connection.

---
```python
class DavidNet(tf.keras.Model):
  def __init__(self, c=64, weight=0.125):
    super().__init__()
    pool = tf.keras.layers.MaxPooling2D()
    self.init_conv_bn = ConvBN(c)
    self.blk1 = ResBlk(c*2, pool, res = True)
    self.blk2 = ResBlk(c*4, pool)
    self.blk3 = ResBlk(c*8, pool, res = True)
    self.pool = tf.keras.layers.GlobalMaxPool2D()
    self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
    self.weight = weight

  def call(self, x, y):
    h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
    h = self.linear(h) * self.weight
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
    loss = tf.reduce_sum(ce)
    correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
    return loss, correct
 ```
 We assemble the DavidNet model using the previously defined classes. In the "init" function, we define all the layers of DavidNet and use the parameters c (no. of channels) and weight (to be multiplied later). We define the variable pool as MaxPooling2D function which will be used in the ResBlk class. In the "call" function, we assemble all the layers and multiply all the weights by 0.125 as done by DavidNet.

DavidNet is a custom ResNet9 model. We pass the first layer through ConvBN with the channels=c, second block through ResBlk with c*2, pooling and a residual connection (res=True), third block through ResBlk with c*4 and pooling, fourth layer is GlobalMaxPool2D and finally passed to a fully connected layer with 10 outputs.

We calculate the crossentropy loss by first calculating it as a tensor and then sum all the elements to give a single number i.e. overall loss. We do not average the loss (as done by dividing it by batch size) as this is logic based on which the entire model is created. The learning rate is scaled down by a factor of 1/512 as the loss is scaled up by the batch size (which is 512 in this case). We calculate the accuracy of the i.e. the total number of correct predictions in a batch (by comparing the predicted and actual class).
    
 ---   
 ```python
 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
len_train, len_test = len(x_train), len(x_test)
y_train = y_train.astype('int64').reshape(len_train)
y_test = y_test.astype('int64').reshape(len_test)

train_mean = np.mean(x_train, axis=(0,1,2))
train_std = np.std(x_train, axis=(0,1,2))

normalize = lambda x: ((x - train_mean) / train_std).astype('float32') # todo: check here
pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

x_train = normalize(pad4(x_train))
x_test = normalize(x_test)
```
We import the train and test data from the CIFAR10 dataset. We normalize the train and test data using the mean and standard deviation of the training data. We pad 4 pixels on all the edges of an image for the training set.

---
```python
model = DavidNet()
batches_per_epoch = len_train//BATCH_SIZE + 1

lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
global_step = tf.train.get_or_create_global_step()
lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE
opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)
```
We call the DavidNet and create an object.

The learning rate is manually scheduled and it is in the shape of a slanted triangle. The learning rate changes in each iteration over a total of 24 epochs. The learning rate is divided by the batch size as explained earlier (becuase the total loss is taken into consideration).

We feed the lr function and momentum to our optimizer.

We have used simple data augmentation technique with random crop of 32x32 and random left/right flip.

---
```python
t = time.time()
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

for epoch in range(EPOCHS):
  train_loss = test_loss = train_acc = test_acc = 0.0
  train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(data_aug).shuffle(len_train).batch(BATCH_SIZE).prefetch(1)

  tf.keras.backend.set_learning_phase(1)
  for (x, y) in tqdm(train_set):
    with tf.GradientTape() as tape:
      loss, correct = model(x, y)

    var = model.trainable_variables
    grads = tape.gradient(loss, var)
    for g, v in zip(grads, var):
      g += v * WEIGHT_DECAY * BATCH_SIZE
    opt.apply_gradients(zip(grads, var), global_step=global_step)

    train_loss += loss.numpy()
    train_acc += correct.numpy()

  tf.keras.backend.set_learning_phase(0)
  for (x, y) in test_set:
    loss, correct = model(x, y)
    test_loss += loss.numpy()
    test_acc += correct.numpy()
 ```
 tf.data.Dataset provides descriptive and efficient input pipelines. It is a simple way to create a dataset.

We iterate over the data for the epochs defined.

We apply dataset transformations on the training dataset and shuffle to preprocess the data.

We start with learning phase = 1, which is for the train data.

We iterate our train variables in tqdm notebook which displays a smart progress bar while training.

tf.GradientTape() records operations for automatic differentiation which is the back propagation step. The variables are fed to the model and the model gives loss and accuracy as output. We take the trainable variables and gradients of the model and apply weight decay. The normal code for applying weight decay (using MomentumWOptimizer) to learning rate in the lr function itself does not work. We also cannot simply change the weight decay parameter to WEIGHT_DECAY*LEARNING_RATE because the learning rate is not constant while training and changes in each iteration. Hence, we add the weight decay factor directly to our gradients.

Then we ask the optimizer to apply the processed gradients.

The train loss and train accuracy is updated with each iteration and recorded.

We follow this procedure for test set as well to get the test loss and test accuracy.

The loss and accuracy is printed for each epoch.

## Finally, we obtain a validation accuracy of 93% using DavidNet.
