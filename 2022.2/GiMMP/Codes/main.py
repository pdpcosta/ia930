# https://keras.io/examples/#computer-vision
## https://keras.io/examples/vision/image_classification_from_scratch/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc
from itertools import cycle


image_size = (48, 48)
batch_size = 32
num_classes = 7

callbacks = [
    keras.callbacks.ModelCheckpoint("checkpoint\\save_at_{epoch}.h5")
]

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)


def visualize_data():
	plt.figure(figsize=(10, 10))
	for images, labels in train_ds.take(1):
	    for i in range(9):
	        ax = plt.subplot(3, 3, i + 1)
	        plt.imshow(images[i].numpy().astype("uint8"))
	        plt.title(int(labels[i]))
	        plt.axis("off")
	plt.show()

def load_images():
    # load images
    train_ds = preprocessing.image_dataset_from_directory(
        # "FER2013/train",
        "ckplus/ck/CK+48/train",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    val_ds = preprocessing.image_dataset_from_directory(
        # "FER2013/test",
        "ckplus/ck/CK+48/validation",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        labels="inferred"
    )
    return train_ds, val_ds

def make_model(input_shape, num_classes, augment_data):
    global data_augmentation

    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) if augment_data else keras.Sequential()(inputs)

    # x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(6, 5, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(16, 5, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    return keras.Model(inputs, outputs)

# deprecated
def make_model_2(input_shape, num_classes, augment_data):
    global data_augmentation

    inputs = keras.Input(shape=input_shape)

    dropout_val = 0.05

    x = data_augmentation(inputs) if augment_data else keras.Sequential()(inputs)

    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_val)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_val)(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_val)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_val)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(dropout_val)(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_val)(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(dropout_val)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def set_model_compile(model):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        # optimizer="RMSprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    # model.summary()
    # plot_model(model, to_file='model.png')
    return model

def set_model_fit(model, train_ds, epochs, val_ds):
    global callbacks
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        callbacks=callbacks, 
        validation_data=val_ds,
        workers=4,
        use_multiprocessing=True)
    return model, history

def predict_on_target(model, path, image_size):
    img = keras.preprocessing.image.load_img(path, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    index_max = np.argmax(predictions[0])
    emotion_map = {
        0: 'Angry',
        1: 'Contempt',
        2: 'Disgust',
        3: 'Fear',
        4: 'Happy',
        5: 'Sad',
        6: 'Surprise'
    }
    return emotion_map.get(index_max), predictions

def restore_model(path):
    if path:
        # return tf.saved_model.load(path)
        return tf.keras.models.load_model(path)

def save_model(model, path):
    if path:
        # tf.saved_model.save(model, path)
        tf.keras.models.save_model(model, path)

    # score = predictions[0]

    # print(predictions[0])
    # print(predictions)

def map_emotion_genre():
    m = {

         'Angry': ['Drama', 'Thriller', 'Action','Romance','Adventure','Crime',
                     'Science','Fiction','Horror','Family','Fantasy','Mystery',
                     'Animation', 'History','Music','War'],

         # ? Equal Sad to avoid error
         'Contempt': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                 'Crime','Science','Fiction','Horror','Family','Fantasy',
                 'Mystery','Animation','History','Music','War'],

         'Disgust': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                     'Crime','Science','Fiction','Horror','Family','Fantasy',
                     'Mystery','Animation','History','Music','War','Documentary'],

         # ? Equal Sad to avoid error
         'Fear': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                 'Crime','Science','Fiction','Horror','Family','Fantasy',
                 'Mystery','Animation','History','Music','War'],

         'Happy': ['Comedy','Thriller','Action','Romance','Adventure','Crime',
                   'Science','Fiction','Horror','Family','Fantasy','Mystery',
                   'Animation','History','Music','War','Documentary','Western',
                   'Foreign','TV','Movie'],

         'Sad': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                 'Crime','Science','Fiction','Horror','Family','Fantasy',
                 'Mystery','Animation','History','Music','War'],

         # ? Equal Happy to avoid error
         'Surprise': ['Comedy','Thriller','Action','Romance','Adventure','Crime',
                   'Science','Fiction','Horror','Family','Fantasy','Mystery',
                   'Animation','History','Music','War','Documentary','Western',
                   'Foreign','TV','Movie'],

         # 'Neutral': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
         #             'Crime','Science','Fiction','Horror','Family','Fantasy',
         #             'Mystery','Animation','History','Music','War','Documentary',
         #             'Western','Foreign','TV','Movie']
         }

    return m

def give_me_movies(n_movies, emotion):
    
    # todos os gêneros que uma emoção "aceita"
    all_genres = map_emotion_genre().get(emotion)
    # escolhe aleatoriamente os gêneros
    picked = np.random.choice(len(all_genres), n_movies)
    # nome do gênero escolhido
    picked_str = [all_genres[x] for x in picked]
    # carrega a base de dados TMDB 5000 (apenas o gênero e o título do filme)
    tmdb = pd.read_csv('tmdb_5000_movies.csv', usecols=['genres', 'title'])
    # lista de gêneros dos filmes
    genres = tmdb.get('genres').to_list()
    # lista de títulos dos filmes
    titles = tmdb.get('title').to_list()
    tmdb = tuple(zip(genres, titles))

    movies = []
    for picked_genre in picked_str:
        picked_tuples = [x for x in tmdb if picked_genre in x[0]]
        index = np.random.choice(len(picked_tuples), 1)
        while picked_tuples[index[0]] in movies:
            index = np.random.choice(len(picked_tuples), 1)
        movies.append(picked_tuples[index[0]])

    return movies

def plot_roc_curve(model, val_ds):
    final_label = list(map_emotion_genre().keys())
    new_class = len(final_label)

    y_pred = np.array([], dtype=np.int)
    y_test =  np.array([], dtype=np.int)
    for x, y in val_ds:
        y_pred = np.concatenate([y_pred, np.argmax(model.predict(x), axis=-1)])
        y_test = np.concatenate([y_test, np.argmax(y.numpy(), axis=-1)])    

    y_pred = get_matrix(y_pred)
    y_test = get_matrix(y_test)

    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(new_class):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['red', 'green','black','blue', 'yellow','purple','orange'])
    return new_class, colors, fpr, tpr, lw, final_label

def get_matrix(indexes):
    indexesT = indexes.T
    a = np.zeros((len(indexes), 7))
    a[range(len(indexes)), indexesT] = 1
    return a
