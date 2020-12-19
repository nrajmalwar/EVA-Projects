## End Game

#### TD3 Algorithm Reinforcement Learning

We use TD3 to train a car to move on a restricted road without using sensors

Logfile-
    - total timesteps, previous state, current sates, action , reward, episode over in each iteration 
  	- total_timesteps, episode count, avg reward over the episode after finishing episode

### State Space

The state space consists of a cropped image of the map along with the superimposed image of the car with the direction towards which it is pointing, its orientation with respect to the goal and the distance moved towards the goal.

### Action Space

It is the angle by which the car turns on the road and attempts to reach the goal. Its range is between -3 and 3.

### Reward Policy

- on outside road -7
- on road -1.7
- on road, and reducing goal distance +0.6
- boundary -15
- destination +110

#### Model For Actor and Critic

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 10)        100       
_________________________________________________________________
dropout_1 (Dropout)          (None, 26, 26, 10)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 26, 26, 10)        40        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 16)        1456      
_________________________________________________________________
dropout_2 (Dropout)          (None, 24, 24, 16)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 24, 24, 16)        64        
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 22, 22, 24)        3480      
_________________________________________________________________
dropout_3 (Dropout)          (None, 22, 22, 24)        0         
_________________________________________________________________
batch_normalization_3 (Batch (None, 22, 22, 24)        96        
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 11, 11, 24)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 11, 10)        250       
_________________________________________________________________
batch_normalization_4 (Batch (None, 11, 11, 10)        40        
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 9, 9, 10)          910       
_________________________________________________________________
dropout_4 (Dropout)          (None, 9, 9, 10)          0         
_________________________________________________________________
batch_normalization_5 (Batch (None, 9, 9, 10)          40        
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 7, 7, 16)          1456      
_________________________________________________________________
dropout_5 (Dropout)          (None, 7, 7, 16)          0         
_________________________________________________________________
batch_normalization_6 (Batch (None, 7, 7, 16)          64        
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 5, 5, 16)          2320      
_________________________________________________________________
```

The final convolution layer output is passed through the Adaptive GAP layer and the orientation and the goal distance is concatenated with the output and finally passed to a series of fully connected layers for final training.
