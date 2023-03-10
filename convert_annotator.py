import os
import logging
import numpy as np
import itertools

from encoder import Encoder
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_datasets as tfds


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH", None)
CACHE_DIR = os.environ.get("CACHE_DIR", None)


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self, list_examples, shuffle=True):
        self.list_examples = list_examples
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_examples)

    def __getitem__(self, index):
        pos = self.indexes[index]
        premise, hypothesis, label = self.__data_generation(self.list_examples[pos])

        return [premise, hypothesis], label

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_examples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_path):
        premise = np.load(file_path)['arr_0'][0]
        hypothesis = np.load(file_path)['arr_0'][1]
        label = np.load(file_path)['arr_1']
        label = l.reshape((len(label), 1))
        return premise, hypothesis, label


class ConveRTAnnotator:
    def __init__(self):
        self.encoder = Encoder()

        try:
            self.model_path = TRAINED_MODEL_PATH
        except:
            self.__prepare_data()
            self.__create_model()
            self.__train_model()

    def __prepare_data(self):
        snli_dataset = tfds.text.Snli()
        snli_dataset.download_and_prepare(download_dir=CACHE_DIR)

        datasets = snli_dataset.as_dataset()
        train_dataset, test_dataset, val_dataset = datasets['train'], datasets['test'], datasets['validation']
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        common_path = CACHE_DIR + '/data'
        val_path = common_path + '/validation'
        test_path = common_path + '/test'
        train_path = common_path + '/train'
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        self.__vectorize_data(val_path+'/val_', val_dataset)
        self.__vectorize_data(test_path+'/test_', test_dataset)
        self.__vectorize_data(train_path+'/train_', train_dataset)

        train_examples = os.listdir(train_path)
        train_examples = [train_path + f_name for f_name in train_examples]
        test_examples = os.listdir(test_path)
        test_examples = [test_path + f_name for f_name in test_examples]
        val_examples = os.listdir(val_path)
        val_examples = [val_path + f_name for f_name in val_examples]

        self.train_generator = DataGenerator(train_examples)
        self.test_generator = DataGenerator(test_examples)
        self.val_generator = DataGenerator(val_examples)

        logger.info(f"All datasets are made.")

    def __vectorize_data(self, data_path, dataset):
        counter = 0
        logger.info(f"Started making {data_path[-4:-1]} dataset.")
        for example in tfds.as_numpy(dataset):
            counter += 1
            premise, hypothesis, label = example['premise'], example['hypothesis'], example['label']

            useless_pos = np.where(label == -1)[0]
            premise = np.delete(premise, useless_pos)
            hypothesis = np.delete(hypothesis, useless_pos)
            label = np.delete(label, useless_pos)

            premise_encoded = self.encoder.encode_sentences(premise)
            hypothesis_encoded = self.encoder.encode_sentences(hypothesis)
            np.savez(data_path+str(counter), [premise_encoded, hypothesis_encoded], label)

            if counter % self.log_freq == 0:
                logger.info(f"Prepared {counter} files.")
        logger.info(f"Prepared all files.")

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
                                 callbacks=[model_checkpoint, csv_logger, early_stopping])
        
        self.model.save(CACHE_DIR + '/model.h5')
        self.model_path = CACHE_DIR + '/model.h5'
    
    def candidate_selection(self, vectorized_history, candidates, threshold=0.8):
        self.model = tf.keras.models.load_model(self.model_path)
        labels = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
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
        return self.encoder.encode_sentences(responses)

    def response_encoding(self, response):
        encoded_response = self.encoder.encode_sentences(response)[0]
        return encoded_response.tolist()
