from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


#----------------------------------------------------------------
## Funcional API
#----------------------------------------------------------------
def get_functional_api_model(config):

    inputs = layers.Input(shape=config.input_shape)
    x = layers.BatchNormalization()(inputs)
    
    # Block One
    x = layers.Conv2D(filters=config.num_filters1, kernel_size=config.kernel_size1, strides=2, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Block Two
    x = layers.Conv2D(filters=config.num_filters2, kernel_size=config.kernel_size2, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)

    # Block Three with Residual Connection
    residual = x  
    x = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense Layers with Dropout
    x = layers.Dense(128, activation=config.activation)(x)
    x = layers.Dropout(0.5)(x)
    
    # Output Layer
    outputs = layers.Dense(config.num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
#----------------------------------------------------------------
## Sequential API
#----------------------------------------------------------------
def get_sequential_api_model(config):
    #print(config.input_shape)
    model = tf.keras.Sequential([
 
        layers.Input(shape=config.input_shape),
        layers.BatchNormalization(),
        
        # Block One
        layers.Conv2D(filters=config.num_filters1, kernel_size=config.kernel_size1, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.Activation(config.activation),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Dropout(0.3),
        
        # Block Two
        layers.Conv2D(filters=config.num_filters2, kernel_size=config.kernel_size2, padding='valid'),
        layers.BatchNormalization(),
        layers.Activation(config.activation),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Dropout(0.3),

        # Block Three (without residual connection)
        layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid'),
        layers.BatchNormalization(),
        layers.Activation(config.activation),
        layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid'),
        layers.BatchNormalization(),
        layers.Activation(config.activation),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Dropout(0.4),

        # Global Average Pooling instead of Flatten
        layers.GlobalAveragePooling2D(),
        
        # Dense Layers with Dropout
        layers.Dense(128, activation=config.activation),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(config.num_classes, activation='softmax')
    ])
    
    return model

#----------------------------------------------------------------
## SubClassing
#----------------------------------------------------------------
class CustomModel(Model):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        
        # Initial batch normalization layer
        self.batch_norm_initial = layers.BatchNormalization()
        
        # Block One
        self.conv1 = layers.Conv2D(filters=config.num_filters1, kernel_size=config.kernel_size1, strides=2, padding='valid')
        self.batch_norm1 = layers.BatchNormalization()
        self.activation1 = layers.Activation(config.activation)
        self.max_pool1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.dropout1 = layers.Dropout(0.2)
        
        # Block Two
        self.conv2 = layers.Conv2D(filters=config.num_filters2, kernel_size=config.kernel_size2, padding='valid')
        self.batch_norm2 = layers.BatchNormalization()
        self.activation2 = layers.Activation(config.activation)
        self.max_pool2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.dropout2 = layers.Dropout(0.2)
        
        # Block Three
        self.conv3a = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid')
        self.batch_norm3a = layers.BatchNormalization()
        self.activation3a = layers.Activation(config.activation)
        self.conv3b = layers.Conv2D(filters=config.num_filters3, kernel_size=config.kernel_size3, padding='valid')
        self.batch_norm3b = layers.BatchNormalization()
        self.activation3b = layers.Activation(config.activation)
        self.max_pool3 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.dropout3 = layers.Dropout(0.4)
        
        # Global Average Pooling
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        
        # Dense Layers with Dropout
        self.dense1 = layers.Dense(128, activation=config.activation)
        self.dropout4 = layers.Dropout(0.5)
        
        # Output Layer
        self.output_layer = layers.Dense(config.num_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        # Initial batch normalization
        x = self.batch_norm_initial(inputs, training=training)
        
        # Block One
        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.activation1(x)
        x = self.max_pool1(x)
        x = self.dropout1(x, training=training)
        
        # Block Two
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)
        x = self.activation2(x)
        x = self.max_pool2(x)
        x = self.dropout2(x, training=training)
        
        # Block Three
        x = self.conv3a(x)
        x = self.batch_norm3a(x, training=training)
        x = self.activation3a(x)
        x = self.conv3b(x)
        x = self.batch_norm3b(x, training=training)
        x = self.activation3b(x)
        x = self.max_pool3(x)
        x = self.dropout3(x, training=training)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        
        # Dense Layers with Dropout
        x = self.dense1(x)
        x = self.dropout4(x, training=training)
        
        # Output Layer
        return self.output_layer(x)
    

#----------------------------------------------------------------
## Hybrid
#----------------------------------------------------------------
def get_hybrid_model(config):
    # Sequential part for Block One
    block1 = tf.keras.Sequential([
        layers.Conv2D(config.num_filters1, kernel_size=config.kernel_size1, strides=2, padding='valid'),
        layers.BatchNormalization(),
        layers.Activation(config.activation),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Dropout(0.2)
    ])
    
    # Sequential part for Block Two
    block2 = tf.keras.Sequential([
        layers.Conv2D(config.num_filters2, kernel_size=config.kernel_size2, padding='valid'),
        layers.BatchNormalization(),
        layers.Activation(config.activation),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.Dropout(0.2)
    ])
    
    # Functional API to create inputs and assemble the model
    inputs = layers.Input(shape=config.input_shape)
    
    x = layers.BatchNormalization()(inputs)
    
    # Use the Sequential blocks within the Functional model
    x = block1(x)
    x = block2(x)

    # Block Three (Functional style for flexibility with skip connections)
    residual = x
    x = layers.Conv2D(config.num_filters3, kernel_size=config.kernel_size3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(config.activation)(x)
    x = layers.Conv2D(config.num_filters3, kernel_size=config.kernel_size3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # Residual connection
    x = layers.Activation(config.activation)(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Head (Dense layers)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=config.activation)(x)
    x = layers.Dropout(0.5)(x)
    
    # Output Layer
    outputs = layers.Dense(config.num_classes, activation='softmax')(x)
    
    # Assemble model
    model = Model(inputs, outputs)
    
    return model

#----------------------------------------------------------------
## AlexNet
#----------------------------------------------------------------
def get_alexnet_model(config):
    # AlexNet input size: (224, 224, 1)
    inputs = layers.Input(shape=(config.img_size, config.img_size, 1)) 

    # The first convolutional layer: 96 filters, kernel 11x11, stride 4
    x = layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'
                      , padding='valid')(inputs)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x) # pool size 3x3, stride 2

    # The second convolutional layer: 256 filters, kernel 5x5
    x = layers.Conv2D(filters=256, kernel_size=5, activation='relu'
                      , padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x) # pool size 3x3, stride 2

    # The third convolutional layer: 384 filters, kernel 3x3
    x = layers.Conv2D(filters=384, kernel_size=3, activation='relu'
                      , padding='valid')(x)

    # The fourth convolutional layer: 384 filters, kernel 3x3
    x = layers.Conv2D(filters=384, kernel_size=3, activation='relu'
                      , padding='valid')(x)

    # The fifth convolutional layer: 256 filters, kernel 3x3
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu'
                      , padding='valid')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # pool size 3x3, stride 2

    # Flatten the exit and pass through dense layers (Flatten)
    x = layers.Flatten()(x)

    # Dense layer (Fully Connected): 4096 units
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout: 50% to reduce overfitting

    # Second Dense layer (Fully Connected): 4096 units
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout: 50% to reduce overfitting

    # Output Layer: softmax, num_classes outputs
    output = layers.Dense(config.num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=output)

    return model


#----------------------------------------------------------------
## Pre-Trained
#----------------------------------------------------------------
#ResNet
def get_preTrained_ResNet(config):
    base_model = tf.keras.applications.ResNet152V2(
        weights='imagenet',
        input_shape=(config.img_size, config.img_size, 3),
        include_top=False)

    base_model.trainable = True

    # Freeze some layers in the base model
    for layer in base_model.layers[:-13]:
        layer.trainable = False

    model = tf.keras.Sequential([
        layers.Input(shape=(config.img_size, config.img_size, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation=config.activation),
        layers.Dropout(0.5),
        layers.Dense(config.num_classes, activation='softmax')
    ])

    return model

#EfficientNet
def get_preTrained_EfficientNet(config):

    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(config.img_size, config.img_size, 3))
    
    base_model.trainable = True

    # Freeze some layers in the base model
    for layer in base_model.layers[:10]:
        layer.trainable = False
    
    inputs = layers.Input(shape=(config.img_size, config.img_size, 3))
    x = base_model(inputs) #, training=False
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=config.activation)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(config.num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model

#VGG16
def get_preTrained_VGG16(config):

    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(config.img_size, config.img_size, 3),
        classes=config.num_classes,
        classifier_activation='softmax'
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation=config.activation),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(config.num_classes, activation='softmax')
    ])

    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True
    
    return model
