import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib

SEQUENCE_LEN = 24


# Replace Lambda with a proper custom layer — safe to save/load
class SumOverTime(layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)


def attention_layer(inputs):
    score   = layers.Dense(1, activation="tanh")(inputs)
    weights = layers.Softmax(axis=1)(score)
    context = layers.Multiply()([inputs, weights])
    output  = SumOverTime()(context)
    return output


def build_model(n_features: int, units: int = 64) -> Model:
    inp = layers.Input(shape=(SEQUENCE_LEN, n_features))

    x = layers.Bidirectional(
        layers.SimpleRNN(units, return_sequences=True)
    )(inp)

    x   = attention_layer(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = Model(inputs=inp, outputs=out, name="BiSRU_Attention")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    selected = joblib.load("models/selected_features.pkl")
    model    = build_model(n_features=len(selected))
    model.summary()