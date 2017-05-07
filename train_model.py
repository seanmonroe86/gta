import numpy as np
from alexnet import alexnet

WIDTH = 80
HEIGHT = 64
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('fit_data.npy')

train = train_data[:-30000]
test = train_data[-30000:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
y = [i[1] for i in train]

test_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

model.fit({'input': X},
          {'targets': y},
          n_epoch=EPOCHS,
          validation_set=(
              {'input': test_X},
              {'targets': test_y}),
          snapshot_step=500,
          show_metric=True,
          run_id=MODEL_NAME)

# tensorboard --logdir=foo:C:/Users/Sean/Desktop/gta/log

model.save(MODEL_NAME)

