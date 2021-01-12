import numpy as np
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import os
import argparse
from model import *
from setup import *
# from keras.utils.visualize_util import plot
# import matplotlib.pyplot as plt


seq_length = 26
embedding_dim = 300
# K = 1000

data_img = 'data/data_img.h5'
data_prepo = 'data/data_prepro.h5'
data_prepo_meta = 'data/data_prepro.json'
embedding_matrix_filename = 'data/embeddings_%s.h5'%embedding_dim
glove_path = 'data/glove_wordvectors/glove.6B.300d.txt'
train_questions_path = 'data/MultipleChoice_mscoco_train2014_questions.json'
val_annotations_path = 'data/mscoco_val2014_annotations.json'
my_model_weights_filename = 'model_weights.h5'
model_weights_filename = 'data/model_weights.h5'

def get_model(dropout_rate, model_weights_filename):
    print("Creating Model...")
    meta_data = metadata()
    num_classes = len(meta_data['ix_to_ans'].keys())
    num_words = len(meta_data['word_to_ix'].keys())
    print(num_classes)
    print(num_words)
    embedding_matrix = prepare_embeddings()
    model = vqa_model(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate, num_classes)
    # plot(model, to_file='model_vqa.png')

    return model


def train(args):
    print('train......................')
    dropout_rate = 0.5
    train_X, train_y = prepare_train(args.data_limit)    
    model = get_model(dropout_rate, model_weights_filename)
    checkpointer = ModelCheckpoint(filepath=my_model_weights_filename, verbose=1)
    history = model.fit(train_X, train_y, epochs=args.epoch, batch_size=args.batch_size, callbacks=[checkpointer], shuffle="batch")
    model.save_weights(model_weights_filename, overwrite=True)
    model.save("out.hdf5")

def val():
    print('eval......................')
    val_X, val_y, multi_val_y = get_val_data()
    model = get_model(0.0, model_weights_filename)
    print("Accuracy validation:", model.evaluate(val_X, val_y))

    # Comparing prediction against multiple choice answers
    true_positive = 0
    preds = model.predict(val_X)
    pred_classes = [np.argmax(_) for _ in preds]
    for i,_ in enumerate(pred_classes):
        if _ in multi_val_y[i]:
            true_positive += 1
    print(np.float(true_positive)/len(pred_classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_limit', type=int, default=215359)
    args = parser.parse_args()

    if args.type == 'train':
        train(args)
    elif args.type == 'val':
        val()

    print('finisheeeed............^_^.............')



    ##################