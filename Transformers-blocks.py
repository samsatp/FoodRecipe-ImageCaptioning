import tensorflow as tf
import numpy as np

## POSITIONAL ENCODER
''' Note:
    This is the way to inject positional information to the input.
    If we don't do this, embeddeding vector of a word will be the same regardless of its postion in sequence.
    - Embeddings represent a token in a d-dimensional space 
      where tokens with similar meaning will be closer to each other.
    - After adding the positional encoding, tokens will be closer to each other 
      based on the similarity of their meaning and their position in the sentence.
'''
def _get_angle(pos, i, d_model):
    angle = 1/np.power(10_000, (2*(i//2))/np.float32(d_model))
    return pos * angle
def positional_encoder(pos, d_model):
    angle = _get_angle(
        pos = np.arange(pos)[:, tf.newaxis],
        i   = np.arange(d_model)[tf.newaxis, :],
        d_model = d_model
    )

    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])

    position_encoded = angle[tf.newaxis, ...]

    return tf.cast(position_encoded, dtype=tf.float32)

## MASKING
''' Note:
    There're 2 things needed to be masked.
    1. Padding tokens.
    2. Masking to prevent look ahead.
'''
def get_mask_padding(seq):
    pad = tf.cast(tf.math.equal(seq,0), tf.float32)

    # Padding mask must be in the dimension that 
    # will be fed into each attention head
    return pad[:, tf.newaxis, tf.newaxis, :]
def get_mask_lookAhead(size):
    # At each time, the model should not see the future outputs
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

## ATTENTION
''' Note:
    In traditional seq2seq model, the decoder use information only from 
    encoder's last hidden state to attend to the input sequence.
        - This make it hard because the information is compressed into 
          a fixed length vector.
    In stead of using only the last one, attention is the way to find the 
    weighted average over all encoder states. This gives an ability for decoder
    to adaptively look for useful information in particular area of the 
    input sequence. And these weights are learnable.
'''
def scaled_dot_product_attention(q, k, v, mask):

    attentionWeights = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(k.shape[-1], tf.float32)
    scaled_attentionWeights = attentionWeights/tf.math.sqrt(dk)

    if mask is not None: scaled_attentionWeights += (mask * -1e9)
    
    scaled_attentionWeights = tf.nn.softmax(scaled_attentionWeights)
    attn_output = tf.matmul(scaled_attentionWeights, v)

    return attn_output, scaled_attentionWeights
def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn.numpy())
    print('Output is:')
    print(temp_out.numpy())


## MULTIHEAD-ATTENTION
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model//num_heads
        assert d_model%num_heads==0

        self.wq = tf.keras.layers.Dense(units=d_model)
        self.wk = tf.keras.layers.Dense(units=d_model)
        self.wv = tf.keras.layers.Dense(units=d_model)

        self.last_dense = tf.keras.layers.Dense(units=d_model)

    def _split_head(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, q, k, v, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self._split_head(q, batch_size)
        k = self._split_head(k, batch_size)
        v = self._split_head(v, batch_size)

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        attn_output = tf.transpose(attn_output, perm=[0,2,1,3])
        attn_output = tf.reshape(attn_output, (batch_size, -1, self.d_model))

        output = self.last_dense(attn_output)

        return output, attn_weights
        
## POINTWISE-FFN
def pointwise_ffn(d_model, dff):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)   # (batch_size, seq_len, d_model)
    ])

## ENCODER LAYER
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = pointwise_ffn(d_model, dff)

        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropOut1 = tf.keras.layers.Dropout(drop_rate)
        self.dropOut2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, seq, mask, isTraining):
        attn_output, attn_weights = self.mha(seq, seq, seq, mask)
        attn_output = self.dropOut1(attn_output, training=isTraining)
        out1 = self.layerNorm1(seq + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropOut2(ffn_output, training=isTraining)
        out2 = self.layerNorm2(out1 + ffn_output)   # (batch_size, input_seq_len, d_model)

        return out2

## DECODER LAYER
''' Note:
    At decoder's second MHA, this is the place that output seq
    attend to input seq to get relavant information. We can use the
    attention weight matrix of this step to visualize how they
    attended to one another.
'''
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(num_heads, d_model)
        self.mha2 = MultiHeadAttention(num_heads, d_model)

        self.ffn = pointwise_ffn(d_model, dff)

        self.layerNorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropOut1 = tf.keras.layers.Dropout(drop_rate)
        self.dropOut2 = tf.keras.layers.Dropout(drop_rate)
        self.dropOut3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, seq, enc_output, mask_lookAhead, mask_padding, isTraining):

        attn1, attn_weight1 = self.mha1(seq, seq, seq, mask_lookAhead)
        attn1 = self.dropOut1(attn1, training=isTraining)
        out1 = self.layerNorm1(seq + attn1)

        attn2, attn_weight2 = self.mha2(q=out1, k=enc_output, v=enc_output, mask=mask_padding)
        attn2 = self.dropOut2(attn2, training=isTraining)
        out2 = self.layerNorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropOut3(ffn_output, training=isTraining)
        out3 = self.layerNorm3(ffn_output)   # (batch_size, target_seq_len, d_model)

        return out3, attn_weight1, attn_weight2


## ENCODER
''' Note:
    Encoder consists of
        1. Input embedding
        2. Positional encoder
        3. N encoder layer
    Input sequence goes through an embedding which is summed with the positional encoding. 
    This summation is the input to the encoder layers. 
    Output of the encoders is the input to the decoder.
'''
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, drop_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        ## Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = d_model)
        self.pos_enc = positional_encoder(pos = vocab_size, d_model = d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, seq, isTraining, mask):

        x = self.embedding(seq)  # (batch_size, vocab_size, d_model)
        x+= self.pos_enc

        x = self.dropout(x, training=isTraining)

        for i in range(self.num_layers):
            x = self.enc_layers[i](seq=x, mask=mask, isTraining=isTraining)

        return x  # (batch_size, vocab_size, d_model)


## DECODER
''' Note:
    Decoder consists of
        1. Target embedding
        2. Positional encoder
        3. N decoder layer
    Target sequence goes through an embedding which is summed with the positional encoding. 
    This summation is the input to the decoder layers. 
    Output of the decoder is the input to the final linear layer.
'''
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, drop_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = d_model)
        self.pos_enc = positional_encoder(
            pos = vocab_size,
            d_model = d_model
        )

        self.dec_layer = [
            DecoderLayer(d_model, num_heads, dff)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, seq, enc_output, isTraining, mask_lookAhead, mask_padding):

        x = self.embedding(seq)
        x+= self.pos_enc

        x = self.dropout(x, training=isTraining)

        attn_weights = {}
        for i in range(len(self.dec_layer)):
            x, attn_w1, attn_w2 = self.dec_layer[i](
                seq = x,
                enc_output = enc_output,
                mask_lookAhead = mask_lookAhead,
                mask_padding = mask_padding,
                isTraining = isTraining
            )
            attn_weights[f'attn_w1_layer{i+1}'] = attn_w1
            attn_weights[f'attn_w2_layer{i+1}'] = attn_w2
        
        return x, attn_weights  # x.shape = (batch_size, target_seq_len, d_model)













if __name__ == '__main__':
    

    example_intput_seq = tf.random.uniform((64, 43, 512))
    print(f'example_intput_seq.shape: {example_intput_seq.shape} ')

    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(
        seq = example_intput_seq, 
        isTraining = False, 
        mask = None
        )
    print(f'enc_output.shape: {sample_encoder_layer_output.shape}')  # (batch_size, input_seq_len, d_model)

    example_target_seq = tf.random.uniform((64, 50, 90))
    print(f'example_target_seq.shape: {example_target_seq.shape}')

    sample_decoder_layer = DecoderLayer(90, 9, 1000)
    sample_decoder_layer_output, _, _ = sample_decoder_layer.call(
        example_target_seq, 
        enc_output = sample_encoder_layer_output,
        mask_lookAhead = None, mask_padding = None, isTraining = False
        )
    print(f'dec_output.shape: {sample_decoder_layer_output.shape}')  # (batch_size, input_target_len, d_model)

    ## TEST ENCODER
    vocab_size = 100
    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, vocab_size=vocab_size)
    temp_input = tf.random.uniform((64, vocab_size), dtype=tf.int64, minval=0, maxval=vocab_size)

    sample_encoder_output = sample_encoder(temp_input, isTraining=False, mask=None)

    print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

    ## TEST DECODER
    sample_decoder = Decoder(num_layers=2, d_model=128, num_heads=8,
                         dff=2048, vocab_size=vocab_size)
    temp_input = tf.random.uniform((64, vocab_size), dtype=tf.int64, minval=0, maxval=vocab_size)

    output, attn = sample_decoder(temp_input,
                                enc_output=sample_encoder_output,
                                isTraining=False,
                                mask_lookAhead=None,
                                mask_padding=None)

    print(output.shape, attn['attn_w1_layer2'].shape)