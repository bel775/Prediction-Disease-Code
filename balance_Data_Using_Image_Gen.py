import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

imagegen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def balance_Data(X_UnBalance, Y_UnBalance):
    batch_size_aux = 1
    train_generator = imagegen.flow(
        X_UnBalance, Y_UnBalance,
        batch_size=batch_size_aux,
        shuffle=True
    )

    X_balanced, Y_balanced = [],[]
    for _ in range(4000):
        img, lbl = next(train_generator)
        X_balanced.append(img[0])  # Append the generated image
        Y_balanced.append(lbl[0])
    
    X_balanced = np.array(X_balanced)
    Y_balanced = np.array(Y_balanced)

    X_merged = np.concatenate((X_balanced, X_UnBalance), axis=0)
    Y_merged = np.concatenate((Y_balanced, Y_UnBalance), axis=0)

    return X_merged, Y_merged