import csv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras import backend as K

from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

# 26x34x1 resimleri kullanacağız
height = 26
width = 34
dims = 1


def readCsv():
    with open('veriseti.csv', 'r') as f:
        # scv dosyasını sözlük biçimiyle okuma
        rows = list(csv.DictReader(f))

    # imgs, tüm görüntüleri içeren numpy dizisidir
    # tgs, resimlerin etiketlerini içeren numpy dizisidir
    imgs = np.empty((len(list(rows)), height, width, dims), dtype=np.uint8)
    tgs = np.empty((len(list(rows)), 1))

    for row, i in zip(rows, range(len(rows))):

        # listeyi görüntü formatına geri dönüştürün
        img = row['image']
        img = img.strip('[').strip(']').split(', ')
        im = np.array(img, dtype=np.uint8)
        im = im.reshape((26, 34))
        im = np.expand_dims(im, axis=2)
        imgs[i] = im

        # açma etiketi 1 ve kapatma etiketi 0
        tag = row['state']
        if tag == 'open':
            tgs[i] = 1
        else:
            tgs[i] = 0

    # veri kümesini karıştır
    index = np.random.permutation(imgs.shape[0])
    imgs = imgs[index]
    tgs = tgs[index]

    return imgs, tgs


# recall metriği için hesaplamalar yapılıyor
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# presicion metriği için hesaplamalar yapılıyor
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# f1 score metriği için hesaplamalar yapılıyor
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# CNN Modelinin sinir ağlarını oluştur
def makeModel():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height, width, dims)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(learning_rate = 0.001), loss='binary_crossentropy', metrics = ['accuracy',f1_m,precision_m, recall_m])#metrik hesaplama
    return model


def main():
    xData, yData = readCsv()

    print(xData.shape[0])
    # görüntülerin değerlerini 0 ile 1 arasında ölçeklendir
    xData = xData.astype('float32')
    xData /= 255

    # veri kümesini test ve eğitim olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.2, shuffle=True,
                                                        random_state=100)

    model = makeModel()

    # biraz veri artırma yapıyoruz
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
    )
    datagen.fit(X_train)
    datagen.fit(X_test)
    # modeli eğit
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=datagen.flow(X_test, y_test), steps_per_epoch=len(X_train) / 32, epochs=50)  #

    # modeli kaydet
    model.save('kirpma_model.hdf5')

    # metrikleri ekrana yazdırıyoruz

    """loss, acc, f1, presicion, recall = model.evaluate(X_test, y_test, batch_size=128)
    print("Test sonucu loss :", str(loss)[:4])
    print("Test sonucu accuracy :", str(acc)[:4])
    print("Test sonucu f1 score :", str(f1)[:4])
    print("Test sonucu presicion :", str(presicion)[:4])
    print("Test sonucu recall :", str(recall)[:4])"""

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy degerleri')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Egitim', 'Test'], loc='upper left')


    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss degerleri')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Egitim', 'Test'], loc='upper left')
    plt.show(dpi=100)


if __name__ == '__main__':
    main()
