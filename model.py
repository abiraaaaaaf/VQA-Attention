from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import merge
from keras.layers import *
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
import h5py

SINGLE_ATTENTION_VECTOR = False


def attention_3d_block(inputs, seq_length):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, seq_length))(a) 
    a = Dense(seq_length, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)

    output_attention_mul = multiply([inputs, a_probs])

    return output_attention_mul


def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes):

    print("Creating image model...")
    img_model = Sequential()
    img_model.add(Dense(1024, input_dim=4096, activation='tanh'))

    print("Creating text model...")
    lstm_model = Sequential()
    lstm_model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=False))
    lstm_model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    lstm_model.add(Dropout(dropout_rate))
    lstm_model.add(LSTM(units=512, return_sequences=False))
    # lstm_model.add(Dropout(dropout_rate))
    # lstm_model.add(Dense(1024, activation='tanh'))

    print("Creating attention model...")

    attention = Dense(1, activation='tanh')(lstm_model.output)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(512)(attention)
    attention = Permute([2, 1])(attention)


    # model2 = Sequential()
    # model2.add(Dense(input_dim=embedding_dim, output_dim=seq_length))
    # model2.add(Dropout(dropout_rate))
    # model2.add(Dense(1024, activation='tanh'))
    # model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
    # # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
    # model2.add(RepeatVector(512))
    # model2.add(Permute([2, 1]))

    print("Applying attention to question")
    attended_layer = multiply([attention, lstm_model.output])
    attended_layer = Lambda(lambda xin: K.sum(xin, axis=1))(attended_layer)
    # attended_layer = TimeDistributedMerge('sum')(attended_layer)
    # attended_layer = TimeDistributed(Dense(1, activation='tanh'))(attended_layer)
    attended_layer = Dropout(dropout_rate)(attended_layer)
    attended_layer = Dense(1024, activation='tanh')(attended_layer)
    # att_model.add(multiply([model2, lstm_model]))
    # output_attention_mul = multiply([model2, lstm_model])
    # att_model.add(Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
    # att_model.add(TimeDistributedMerge('sum'))  # Sum the weighted elements.
    att_model = Model(inputs=[lstm_model.input], outputs=[attended_layer])


    print("Merging final model...")
    merged_layers = concatenate([img_model.output, att_model.output])
    merged_out = Dropout(dropout_rate)(merged_layers)
    merged_out = Dense(1000, activation='tanh')(merged_out)
    merged_out = Dropout(dropout_rate)(merged_out)
    out = Dense(num_classes, activation='softmax')(merged_out)
    fc_model = Model(inputs=[img_model.input, att_model.input], outputs=[out])
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return fc_model
