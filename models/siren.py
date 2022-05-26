import tensorflow as tf
import numpy as np
import sys

sys.path.append('')

from models.modulated_linear import ModulatedLinear


class SineLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        use_bias=False,
        is_first=False,
        omega_0=30,
        demodulation=True,
        outermost_linear=False, 
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.is_first = is_first
        self.omega_0 = omega_0
        self.demodulation = demodulation
        self.outermost_linear=outermost_linear
        kernel_initializer = None
        if is_first:
            kernel_initializer = tf.random_uniform_initializer(
                -1/output_dim,
                1/output_dim
            )
        else:
            kernel_initializer = tf.random_uniform_initializer(
                -np.sqrt(6 / output_dim) / omega_0,
                np.sqrt(6 / output_dim) / omega_0
            )
        self.linear = ModulatedLinear(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            demodulation=demodulation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs):
        assert isinstance(inputs, tuple) and len(inputs) == 2
        x, style = inputs
        x = self.linear((x, style))
        if self.outermost_linear:
            return x
        else:
            return tf.math.sin(self.omega_0 * x)


class Siren(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        hidden_layers,
        out_dim,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        demodulation=True,
        outermost_linear=False,
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.out_dim = out_dim
        self.first_omega_0=first_omega_0
        self.hidden_omega_0=hidden_omega_0
        self.demodulation=demodulation
        self.outermost_linear=outermost_linear
        self.net_layers = []
        self.net_layers.append(
            SineLayer(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
                demodulation=demodulation,
            )
        )

        for _ in range(hidden_layers):
            self.net_layers.append(
                SineLayer(
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    demodulation=demodulation,
                )
            )

        self.net_layers.append(
            SineLayer(
                hidden_dim=hidden_dim,
                output_dim=out_dim,
                is_first=False,
                omega_0=hidden_omega_0,
                demodulation=demodulation,
                outermost_linear=outermost_linear,
            )
        )

    def call(self, inputs):
        assert isinstance(inputs, tuple) and len(inputs) == 2
        x, style = inputs
        for layer in self.net_layers:
            x = layer((x, style))
        return x



if __name__ == "__main__":
    emb_dim = 768 # 16*16*3
    layer = Siren(
        hidden_dim=emb_dim,
        hidden_layers=3,
        out_dim=3,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        demodulation=True,
        outermost_linear=False, 
    )
    e_fou = tf.random.uniform([2*196,256,768], dtype=tf.float32)
    y = tf.random.uniform([2,196,768], dtype=tf.float32)
    o1 = layer((e_fou, y))
    tf.print('o1:', tf.shape(o1))