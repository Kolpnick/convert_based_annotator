import os

import numpy as np

import tensorflow as tf
import tensorflow_hub as tfhub

import itertools

from sentence_encoder import SentenceEncoder


TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH", None)
CACHE_DIR = os.environ.get("CACHE_DIR", None)
DATA_DIR = os.environ.get("DATA_DIR", None)


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_examples, shuffle=True):
        self.list_examples = list_examples
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_examples)

    def __getitem__(self, index):
        pos = self.indexes[index]
        p, h, l = self.__data_generation(self.list_examples[pos])

        return [p, h], l

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_path):
        p = np.load(file_path)['arr_0'][0]
        h = np.load(file_path)['arr_0'][1]
        l = np.load(file_path)['arr_1']
        l = l.reshape((len(l), 1))
        return p, h, l


class ConveRTAnnotator:
    def __init__(self):
        if TRAINED_MODEL_PATH:
            self.model = tf.keras.models.load_model(TRAINED_MODEL_PATH)
        else:
            self.__prepare_data()
            self.__create_model()
            self.__train_model()
    
        self.sentence_encoder = SentenceEncoder()

    def __prepare_data(self):
        train_path = DATA_DIR + '/Train/'
        test_path = DATA_DIR + '/Test/'
        val_path = DATA_DIR + '/Validation/'

        train_examples = os.listdir(train_path)
        train_examples = [train_path + f_name for f_name in train_examples]
        test_examples = os.listdir(test_path)
        test_examples = [test_path + f_name for f_name in test_examples]
        val_examples = os.listdir(val_path)
        val_examples = [val_path + f_name for f_name in val_examples]

        self.train_generator = DataGenerator(train_examples)
        self.test_generator = DataGenerator(test_examples)
        self.val_generator = DataGenerator(val_examples)

    def __create_model(self):
        inp_p = tf.keras.layers.Input(shape=(1024))
        inp_h = tf.keras.layers.Input(shape=(1024))
        combined = tf.keras.layers.concatenate([inp_p, inp_h])
        linear_1 = tf.keras.layers.Dense(1024, activation='relu')(combined)
        dropout_1 = tf.keras.layers.Dropout(0.45)(linear_1)
        linear_2 = tf.keras.layers.Dense(512, activation='relu')(dropout_1)
        linear_3 = tf.keras.layers.Dense(256, activation='relu')(linear_2)
        output = tf.keras.layers.Dense(3, activation='softmax')(linear_3)

        self.model = tf.keras.models.Model(inputs=[inp_p, inp_h], outputs=output)
        self.model.compile(
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer='adam',
                    metrics=['accuracy'])
        
    def __train_model(self):
        log_dir = CACHE_DIR + '/logs/'
        ch_path = CACHE_DIR + '/checkpoints/cp-{epoch:04d}.ckpt'
        csv_logger = tf.keras.callbacks.CSVLogger(log_dir + 'log.csv')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                            filepath=ch_path,
                            save_weights_only=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=10)

        history = self.model.fit(x=self.train_generator, 
                                 validation_data=self.val_generator,
                                 use_multiprocessing=True,
                                 workers=6, epochs=100,
                                 callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping])
        
        self.model.save(CACHE_DIR + '/model/model.h5')
    
    def candidate_selection(self, vectorized_history, candidates, threshold=0.8):
        labels = {0: 'no_contradiciton', 1: 'neutral', 2: 'contradiciton'}
        rez_dict = dict(zip(candidates, [{'decision': labels[0], labels[0]: 1.0, labels[1]: 0.0, labels[2]: 0.0}]*len(candidates)))
        if vectorized_history:
            vectorized_candidates = self.__multiple_responses_encoding(candidates)
            combinations = list(itertools.product(vectorized_history, vectorized_candidates))
            history_arr = []
            candidates_arr = []
            for item in combinations:
                history_arr.append(item[0])
                candidates_arr.append(item[1])
            pred_rez = self.model.predict([history_arr, candidates_arr])
            for i in range(len(pred_rez)):
                j = i % len(candidates)
                cand = candidates[j]
                row_probab = pred_rez[i]
                if row_probab[2] < threshold:
                    row_probab[2] = -row_probab[2]
                label = np.argmax(row_probab, axis=-1)
                if rez_dict[cand]['decision'] != 2:
                    rez_dict[cand] = {'decision': labels[label], labels[0]: row_probab[0], labels[1]: row_probab[1], labels[2]: np.abs(row_probab[2])}
            return rez_dict
        else:
            return rez_dict

    def __multiple_responses_encoding(self, responses):
        return self.sentence_encoder.encode_sentences(responses)

    def response_encoding(self, response):
        encoded_response = self.sentence_encoder.encode_sentences(response)[0]
        return encoded_response.tolist()
