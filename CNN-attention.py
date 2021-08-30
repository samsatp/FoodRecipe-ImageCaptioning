import tensorflow as tf

from tensorflow.keras.layers import Conv2D, GRU, Attention, Dense, Embedding, AveragePooling2D, Reshape

class CNN_encoder(tf.keras.Model):
    def __init__(self, w, h, channel, n_cov, filters_list, kernel_sizes, n_pool):
        super().__init__()
        self.w = w
        self.h = h
        self.channel = channel

        assert len(filters_list) == len(kernel_sizes)
        assert n_cov == n_pool
        self.n_cov = len(filters_list)
        self.n_pool = n_pool
        self.filters_list = filters_list
        self.kernel_sizes = kernel_sizes
        self.conv_layers = [
                            Conv2D(filters = self.filters_list[i], kernel_size = self.kernel_sizes[i]) 
                            for i in range(self.n_cov)
                            ]
        self.pooling_layers =[
                              AveragePooling2D(pool_size=(2,2)) for i in range(self.n_pool)
                             ]
    def call(self, image, verbose):

        batch_size= image.shape[0]
        if verbose: print(f'input image: {image.shape}')
        
        for i in range(self.n_cov):
            image = self.conv_layers[i](image)
            if verbose: print(f'conv_{i} output: {image.shape}')
            
            image = self.pooling_layers[i](image)
            if verbose: print(f'pool_{i} output: {image.shape}')
        
        #image = tf.reshape(image, [batch_size, -1, self.filters_list[-1]])
        image = tf.keras.layers.Reshape((-1, self.filters_list[-1]))(image)
        if verbose: print(f'final output: {image.shape}')

        return image        


class CNNattn(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim, gru_units, w_units, head_denses_units,**cnn_encoder_kwargs):
        super().__init__()
        self.cnn_encoder = CNN_encoder(**cnn_encoder_kwargs)
        self.embedding = Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)
        self.gru = GRU(units = gru_units, return_sequences=True)
        
        self.wq = Dense(units=w_units, activation='relu')
        self.wk = Dense(units=w_units, activation='relu')

        self.attention = Attention(use_scale=True)

        self.head_denses = [
                            Dense(unit, activation='relu') for unit in head_denses_units
                           ]
        self.output_dense = Dense(vocab_size)

    def call(self, image, text, verbose):

        output_cnn_enc = self.cnn_encoder(image, verbose)

        emb_vectors = self.embedding(text)
        emb_vectors = tf.expand_dims(emb_vectors, axis=1)
        if verbose: print(f'embed output: {emb_vectors.shape}')
        

        out_sequences = self.gru(emb_vectors)
        if verbose: print(f'GRU output: {out_sequences.shape}')

        wq = self.wq(out_sequences)
        wk = self.wk(output_cnn_enc)

        output = self.attention([wq, wk, wk])
        if verbose: print(f'attention output: {output.shape}')

        for i in range(len(self.head_denses)):
            output = self.head_denses[i](output)
        if verbose: print(f'head_denses output: {output.shape}')
        
        output = self.output_dense(output)
        if verbose: print(f'output: {output.shape}')

        output = tf.reshape(output, shape=(-1, output.shape[-1]))
        if verbose: print(f'output: {output.shape}')

        return output

if __name__ == '__main__':

    batch_size = 32
    w = 64
    h = 64
    channel = 3

    sample_images = tf.random.uniform((batch_size, w, h, channel), minval=0, maxval=1, 
                                    dtype=tf.float32, name='sample_imgs')
    vocab_size = 100
    sample_text = tf.random.uniform((batch_size, vocab_size), dtype=tf.int32, 
                                    maxval=vocab_size)

    sample_dataset = tf.data.Dataset.from_tensor_slices((sample_images, sample_text))
    sample_dataset = sample_dataset.batch(8)

    my_model = CNNattn(
        vocab_size = vocab_size,
        emb_dim = 64,
        gru_units = 16,
        w_units = 64,
        head_denses_units = [100],
        w = 64,
        h = 64,
        channel = 3,
        n_cov = 2,
        filters_list =[48, 48],
        kernel_sizes = [(4,4), (4,4)],
        n_pool = 2
    )

    for img, text in sample_dataset.take(1):
        print(f'imag.shape: {img.shape}')
        print(f'text.shape: {text.shape}\n')
        o = my_model(text=text, image=img, verbose=1)
        break