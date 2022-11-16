# https://keras.io/examples/#computer-vision
## https://keras.io/examples/vision/image_classification_from_scratch/


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import generic_utils


image_size = (48, 48)
batch_size = 32
num_classes = 7

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5")
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
        "FER2013/train",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    val_ds = preprocessing.image_dataset_from_directory(
        "FER2013/test",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    return train_ds, val_ds






# def xxxxxxx(input_shape, num_classes, augment_data):

#     model = Sequential()
#     model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Conv2D(64, (3, 3), activation = 'relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Flatten())
#     model.add(Dense(128, activation = 'relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation = 'softmax'))

#     model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='RMSprop')
    
#     # return model
#     outputs = layers.Dense(num_classes, activation='softmax')(x)
#     return keras.Model(inputs, outputs)




def make_model(input_shape, num_classes, augment_data):
    global data_augmentation

    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) if augment_data else keras.Sequential()(inputs)

    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(7, activation='softmax')(x)

    return keras.Model(inputs, outputs)


def make_model1(input_shape, num_classes, augment_data):
    global data_augmentation


    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs) if augment_data else keras.Sequential()(inputs)

    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256]:#, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = layers.Dropout(0.1)(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def set_model_compile(model):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
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
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Neutral',
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
    m = {'Angry': ['Drama', 'Thriller', 'Action','Romance','Adventure','Crime',
         'Science','Fiction','Horror','Family','Fantasy','Mystery','Animation',
         'History','Music','War'],

         'Disgust': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                     'Crime','Science','Fiction','Horror','Family','Fantasy',
                     'Mystery','Animation','History','Music','War','Documentary'],
         
         # ? Equal neutral to avoid error
         'Fear': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                     'Crime','Science','Fiction','Horror','Family','Fantasy',
                     'Mystery','Animation','History','Music','War','Documentary',
                     'Western','Foreign','TV','Movie'],

         'Happy': ['Comedy','Thriller','Action','Romance','Adventure','Crime',
                   'Science','Fiction','Horror','Family','Fantasy','Mystery',
                   'Animation','History','Music','War','Documentary','Western',
                   'Foreign','TV','Movie'],

         'Sad': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                 'Crime','Science','Fiction','Horror','Family','Fantasy',
                 'Mystery','Animation','History','Music','War'],

         # ? Equal neutral to avoid error
         'Surprise': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                     'Crime','Science','Fiction','Horror','Family','Fantasy',
                     'Mystery','Animation','History','Music','War','Documentary',
                     'Western','Foreign','TV','Movie'],

         'Neutral': ['Drama','Comedy','Thriller','Action','Romance','Adventure',
                     'Crime','Science','Fiction','Horror','Family','Fantasy',
                     'Mystery','Animation','History','Music','War','Documentary',
                     'Western','Foreign','TV','Movie']
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















# if __name__ == "__main__":
#     train_ds, val_ds = load_images()
#     data_augmentation = augmentation()
#     train_ds = train_ds.prefetch(buffer_size=32)
#     val_ds = val_ds.prefetch(buffer_size=32)
#     model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
#     ##keras.utils.plot_model(model, show_shapes=True)
    

    
    
#     img = keras.preprocessing.image.load_img("FER2013/test/disgust/PrivateTest_21629266.jpg", target_size=image_size)
#     img_array = keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#     predictions = model.predict(img_array)
#     score = predictions[0]
#     print("This image is %.2f percent Disgust." % (100 * (1 - score)))
