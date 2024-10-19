from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras.models import Model

def cnn_model_functional_API(config):


    inputs = layers.Input(shape=config.input_shape)

    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(config.num_filters1, (config.kernel_size1, config.kernel_size1), activation=config.activation)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(config.num_filters2, (config.kernel_size2, config.kernel_size2), activation=config.activation)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)

    x = layers.Dense(256)(x)
    x = layers.Dense(256)(x)

    outputs = layers.Dense(config.num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


#filers1 = 16 | filters2=32 | filters3=64 
def get_funcional_api_model(config):

    inputs = layers.Input(shape=config.input_shape)

    # Block One
    x = layers.Conv2D(filters=config.num_filters1, kernel_size=config.kernel_size1, padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block Two
    x = layers.Conv2D(filters=config.num_filters2, kernel_size=config.kernel_size1, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)

    # Block Three
    x = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size1, padding='valid')(x)
    x = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size1, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation=config.activation)(x)
    x = layers.Dropout(0.5)(x)

    #Output
    output = layers.Dense(config.num_classes, activation='softmax')(x)

    model = keras.Model(inputs=[inputs], outputs=output)

    return model

def get_sequential_api_model(config):
    model = models.Sequential()

    # Block One
    model.add(layers.Conv2D(filters=config.num_filters1, kernel_size=config.kernel_size1, padding='valid', input_shape=config.input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(config.activation))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.2))

    # Block Two
    model.add(layers.Conv2D(filters=config.num_filters2, kernel_size=config.kernel_size2, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(config.activation))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.2))

    # Block Three
    model.add(layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid'))
    model.add(layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(config.activation))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.4))

    # Head
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=config.activation))
    model.add(layers.Dropout(0.5))

    # Output
    model.add(layers.Dense(config.num_classes, activation='softmax'))

    return model


class CustomModel(Model):
    def __init__(self,config):
        super(CustomModel, self).__init__()

        # Block One
        self.conv1 = layers.Conv2D(filters=config.num_filters1, kernel_size=config.kernel_size1, padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation(config.activation)
        self.pool1 = layers.MaxPool2D()
        self.drop1 = layers.Dropout(0.2)

        # Block Two
        self.conv2 = layers.Conv2D(filters=config.num_filters2, kernel_size=config.kernel_size2, padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation(config.activation)
        self.pool2 = layers.MaxPool2D()
        self.drop2 = layers.Dropout(0.2)

        # Block Three
        self.conv3_1 = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid')
        self.conv3_2 = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid')
        self.bn3 = layers.BatchNormalization()
        self.act3 = layers.Activation(config.activation)
        self.pool3 = layers.MaxPool2D()
        self.drop3 = layers.Dropout(0.4)

        # Head
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation=config.activation)
        self.drop4 = layers.Dropout(0.5)

        # Final Layer (Output)
        self.output_layer = layers.Dense(config.num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Block One
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        # Block Two
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        # Block Three
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.bn3(x, training=training)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        # Head
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop4(x, training=training)

        # Final Layer (Output)
        return self.output_layer(x)
    

def get_alexnet_model(config):
    inputs = layers.Input(shape=config.input_shape)

    # Primera capa convolucional
    x = layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Segunda capa convolucional
    x = layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Tercera capa convolucional
    x = layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Cuarta capa convolucional
    x = layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Quinta capa convolucional
    x = layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    # Aplanar la salida y pasar a través de capas densas
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Capa de salida
    output = layers.Dense(config.num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=output)

    return model