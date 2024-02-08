from keras import layers, models
from keras.layers import Dropout
from keras.datasets import cifar10

# Caricamento del dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizzazione dei pixel

# Creazione della CNN
def create_cnn(num_conv_layers, conv_size, num_fc_layers, fc_size):

    model = models.Sequential()

    # Aggiunta degli strati convoluzionali
    for _ in range(num_conv_layers):
        model.add(layers.Conv2D(conv_size, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(Dropout(0.5)) # Dropout per limitare l'overfitting

    # Aggiunta degli strati fully-connected
    for _ in range(num_fc_layers):
        model.add(layers.Dense(fc_size, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(Dropout(0.3)) # Dropout per limitare l'overfitting

    model.add(layers.Dense(10, activation='softmax'))  # Strato di output

    return model


# Addestramento della CNN
def train_cnn(model, epochs):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test)) # Addestramento

    _, accuracy = model.evaluate(x_test, y_test) # Test
    return accuracy, history



