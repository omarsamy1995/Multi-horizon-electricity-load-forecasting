"""
Classic Deep Learning Model: Sequence-to-Sequence GRU with Future Calendar Decoder.

Architecture:
- Encoder: Multi-layer GRU processing historical observations (168 timesteps, 9 features)
- Decoder: GRU taking future calendar features (24 timesteps, 8 features) + Bahdanau Temporal Attention
- Quantile Projection Head: Guarantees non-crossing quantiles [q10, q50, q90]
"""

from typing import Tuple, Optional, Union, Dict, Any
import tensorflow as tf

from src.models.custom_layers import NonCrossingQuantileHead


@tf.keras.utils.register_keras_serializable(package="models")
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Additive Attention mechanism (Bahdanau et al.)
    Computes attention alignment between decoder state and encoder memory.
    """
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.w_enc = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="w_enc")
        self.w_dec = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="w_dec")
        self.v = tf.keras.layers.Dense(1, use_bias=False, name="v")

    def call(self, query: tf.Tensor, memory: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            query: Current decoder state (B, 1, hidden_dim)
            memory: All encoder hidden states (B, L_enc, hidden_dim)
        Returns:
            context: (B, 1, hidden_dim)
            weights: (B, 1, L_enc)
        """
        score = self.v(tf.nn.tanh(self.w_enc(memory) + self.w_dec(query)))  # (B, L_enc, 1)
        score = tf.transpose(score, [0, 2, 1])                                # (B, 1, L_enc)
        weights = tf.nn.softmax(score, axis=-1)                               # (B, 1, L_enc)
        context = tf.matmul(weights, memory)                                  # (B, 1, hidden_dim)
        return context, weights

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="models")
class Seq2SeqGRU(tf.keras.Model):
    """
    Seq2Seq GRU with Future Calendar Conditioning and Temporal Attention in tf.keras.
    
    Accepts:
        x_past: Tensor of shape (B, 168, 9)
        x_future_cal: Tensor of shape (B, 24, 8)
    Returns:
        Tensor of shape (B, 24, 3) for quantiles [0.1, 0.5, 0.9]
    """
    def __init__(
        self,
        past_dim: int = 9,
        future_cal_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        name: str = "seq2seq_gru",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.past_dim = past_dim
        self.future_cal_dim = future_cal_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_attention = use_attention

        # Encoder projection & GRU layers
        self.enc_in_proj = tf.keras.layers.Dense(hidden_dim, name="enc_in_proj")
        self.enc_relu = tf.keras.layers.ReLU(name="enc_relu")
        self.enc_drop = tf.keras.layers.Dropout(dropout, name="enc_drop")

        self.encoder_gru_layers = []
        for i in range(num_layers):
            self.encoder_gru_layers.append(
                tf.keras.layers.GRU(
                    hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    name=f"encoder_gru_{i}"
                )
            )

        # Decoder projection
        self.dec_in_proj = tf.keras.layers.Dense(hidden_dim, name="dec_in_proj")
        self.dec_relu = tf.keras.layers.ReLU(name="dec_relu")
        self.dec_drop = tf.keras.layers.Dropout(dropout, name="dec_drop")

        # Decoder RNN Cells for step-by-step decoding
        decoder_cells = [
            tf.keras.layers.GRUCell(hidden_dim, name=f"decoder_gru_cell_{i}")
            for i in range(num_layers)
        ]
        self.decoder_rnn = tf.keras.layers.StackedRNNCells(decoder_cells, name="decoder_stacked_cells")

        if use_attention:
            self.attention = BahdanauAttention(hidden_dim, name="bahdanau_attention")
        else:
            self.attention = None

        # Non-crossing quantile projection head
        self.quantile_head = NonCrossingQuantileHead(output_horizon=None, name="quantile_head")

    def call(
        self,
        inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]],
        x_future_cal: Optional[tf.Tensor] = None,
        return_attention: bool = False,
        training: Optional[bool] = None
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        if isinstance(inputs, (tuple, list)):
            x_past, x_future_cal = inputs[0], inputs[1]
        elif isinstance(inputs, dict):
            x_past = inputs.get("x_past", inputs.get("past"))
            x_future_cal = inputs.get("x_future_cal", inputs.get("future_cal"))
        else:
            x_past = inputs

        B = tf.shape(x_past)[0]
        T_future = tf.shape(x_future_cal)[1]

        # 1. Encode past observations
        enc_in = self.enc_drop(self.enc_relu(self.enc_in_proj(x_past)), training=training)

        h = enc_in
        states = []
        for gru_layer in self.encoder_gru_layers:
            h, state = gru_layer(h, training=training)
            states.append(state)
        enc_out = h

        # 2. Decode future calendar step-by-step
        dec_inputs = self.dec_drop(self.dec_relu(self.dec_in_proj(x_future_cal)), training=training)

        cur_states = states  # list of hidden states per layer
        step_quantiles_list = []
        attn_list = []

        for t in range(24):
            cal_step = dec_inputs[:, t, :]  # (B, hidden_dim)

            if self.use_attention:
                query = tf.expand_dims(cur_states[-1], axis=1)  # (B, 1, hidden_dim)
                context, attn_w = self.attention(query, enc_out)  # context: (B, 1, hidden_dim), attn_w: (B, 1, 168)
                attn_list.append(attn_w)
                context_2d = tf.squeeze(context, axis=1)
                dec_in_step = tf.concat([cal_step, context_2d], axis=-1)  # (B, 2*hidden_dim)
            else:
                dec_in_step = cal_step

            dec_out, cur_states = self.decoder_rnn(dec_in_step, cur_states, training=training)

            if self.use_attention:
                dec_out_3d = tf.expand_dims(dec_out, axis=1)  # (B, 1, hidden_dim)
                head_feat = tf.concat([dec_out_3d, context], axis=-1)  # (B, 1, 2*hidden_dim)
            else:
                head_feat = tf.expand_dims(dec_out, axis=1)  # (B, 1, hidden_dim)

            q_step = self.quantile_head(head_feat)  # (B, 1, 3)
            step_quantiles_list.append(q_step)

        quantiles = tf.concat(step_quantiles_list, axis=1)  # (B, 24, 3)

        if return_attention and self.use_attention:
            attn_all = tf.concat(attn_list, axis=1)  # (B, 24, 168)
            return quantiles, attn_all

        return quantiles

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "past_dim": self.past_dim,
            "future_cal_dim": self.future_cal_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "use_attention": self.use_attention
        })
        return config
