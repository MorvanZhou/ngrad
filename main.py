from tensorflow import keras
import numpy as np
import tensorflow as tf

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3. + 2. + noise

train_x, train_y = data_x[:900], data_y[:900]
test_x, test_y = data_x[900:], data_y[900:]


f1 = keras.models.Sequential([keras.layers.Dense(10, input_dim=1)], name="f1")
f2 = keras.models.Sequential([keras.layers.Dense(10, input_dim=10)], name="f2")
f3 = keras.models.Sequential([keras.layers.Dense(1, input_dim=10)], name="f3")

g3 = keras.models.Sequential([keras.layers.Dense(10, input_dim=1)], name="g3")
g2 = keras.models.Sequential([keras.layers.Dense(10, input_dim=10)], name="g2")


def hidden_target(h, gh, gt):
    _t = h - gh + gt
    return _t


opt = keras.optimizers.SGD(0.01)
mse = keras.losses.MeanSquaredError()


def pred(x):
    return f3(f2(f1(x)))


for t in range(100):
    bi = np.random.randint(0, len(train_x), 12)
    bx, by = train_x[bi], train_y[bi]
    with tf.GradientTape(persistent=True) as tape:
        h1 = f1(bx)
        h2 = f2(h1)
        h3 = f3(h2)

        g3h3 = g3(h3)
        g2h2 = g2(h2)
        t3 = by
        t2 = hidden_target(h2, g3h3, g3(t3))
        t1 = hidden_target(h1, g2h2, g2(t2))

        e3 = mse(t3, h3)
        e2 = mse(t2, h2)
        e1 = mse(t1, h1)

        eb3 = mse(h2, g3h3)
        eb2 = mse(h1, g2h2)
        print(e3.numpy())

    grad_f3 = tape.gradient(e3, f3.trainable_variables)
    grad_f2 = tape.gradient(e2, f2.trainable_variables)
    grad_f1 = tape.gradient(e1, f1.trainable_variables)

    grad_g3 = tape.gradient(eb3, g3.trainable_variables)
    grad_g2 = tape.gradient(eb2, g2.trainable_variables)

    for grad, n in zip([grad_f1, grad_f2, grad_f3, grad_g2, grad_g3], [f1, f2, f3, g2, g3]):
        opt.apply_gradients(zip(grad, n.trainable_variables))


