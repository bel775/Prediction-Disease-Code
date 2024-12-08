from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from blob_generator import generate_new_blob_img

early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    min_delta=1e-7,
                    restore_best_weights=True,
                )

plateau = ReduceLROnPlateau(
              monitor='val_loss',
              factor = 0.2,
              patience = 2,
              min_delt = 1e-7,
              cooldown = 0,
              verbose = 1
          )
"""model_checkpoint = ModelCheckpoint(
        'best_model.h5', 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min',
        save_format='tf')"""

SEED = 42

#----------------------------------------------------------------
## Normal Train
#----------------------------------------------------------------
def train_blob_model(model, param_config):
    N = 6000
    big_blob_size = 8
    big_blob_range = 2
    labels = np.random.randint(3, size = N)
    imgs = np.zeros((N, param_config.img_size, param_config.img_size))

    for i in range(N):
        if labels[i] == 0:
            x = generate_new_blob_img(ellipse=True, size=param_config.img_size,num_big_blobs=4)
        elif labels[i] == 1:
            x = generate_new_blob_img(ellipse = False, size = param_config.img_size, num_big_blobs=4)
        elif labels[i] == 2:
            x = generate_new_blob_img(ellipse= False, size = param_config.img_size, num_big_blobs=0)
        imgs[i,:,:] = x

    X_blob = np.expand_dims(imgs, -1)
    #X_blob = np.transpose(X_blob, (2,1,0,3))
    Y_blob = labels

    print(Y_blob.shape)
    print(X_blob.shape)
    model.fit(X_blob, Y_blob, batch_size=param_config.batch_size, epochs=param_config.epochs, validation_split=0.2)

    return model

def normal_train (model,X_train,Y_train,X_test, Y_test, param_config):

  #model = train_blob_model(model, param_config)

  model.fit(X_train, Y_train, batch_size=param_config.batch_size, epochs=param_config.epochs, 
            validation_split=0.2,callbacks=[early_stopping], verbose=2) #callbacks=[early_stopping, model_checkpoint]

  loss, acc = model.evaluate(X_test,Y_test,batch_size=param_config.batch_size, verbose=0)
  print(f'loss: {loss:.4f} acc: {acc:.4f}')

  y_pred = model.predict(X_test)
  y_pred_normalized = y_pred / np.sum(y_pred, axis=1, keepdims=True)

  roc_auc = roc_auc_score(Y_test, y_pred_normalized, multi_class='ovr')
  print(f'AUC {roc_auc:.4f}')

  #y_pred_int = y_pred_normalized.argmax(axis=1)
  return model, roc_auc, acc, loss


#----------------------------------------------------------------
## Data Generator
#----------------------------------------------------------------

datagen = ImageDataGenerator(
        #rescale=1/255.,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        #shear_range=0.15,
        #horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2)

datagen_test = ImageDataGenerator() #rescale=1/255.

def image_data_generator (model,X_train,Y_train,X_test, Y_test, param_config):


    ds_train = datagen.flow(X_train, Y_train, batch_size=param_config.batch_size, subset='training',seed=SEED) #,target_size=(64, 64) | class_mode='categorical' | ,seed=SEED
    ds_val = datagen.flow(X_train, Y_train, batch_size=param_config.batch_size, subset='validation',seed=SEED) #,target_size=(64, 64) | class_mode='categorical' | ,seed=SEED
    ds_test = datagen_test.flow(X_test,Y_test,batch_size = 1,shuffle = False)

    #keras.backend.clear_session()
    #model = get_funcional_api_model(param_config)
    #model.compile(loss='sparse_categorical_crossentropy',
    #              optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    #              metrics=['accuracy'])
    

    steps_per_epoch = (len(X_train)*0.8)//param_config.batch_size
    validation_steps = (len(X_train)*0.2)//param_config.batch_size

    history = model.fit(ds_train,
                      #batch_size = param_config.batch_size,
                      epochs = param_config.epochs,
                      validation_data=ds_val,
                      callbacks=[early_stopping, plateau],
                      steps_per_epoch= steps_per_epoch, # (len(X_train) / batch_size)
                      validation_steps= validation_steps); #(len(X_val) / batch_size)

    score = model.evaluate(ds_val, steps = len(ds_val), verbose = 0)

    #print('Val loss:', score[0])
    #print('Val accuracy:', score[1])

    score = model.evaluate(ds_test, steps = len(ds_test), verbose = 0)

    print('loss:', score[0])
    print('accuracy:', score[1])

    Y_test_pred = model.predict(ds_test, steps=len(ds_test))
    #print(len(np.unique(Y_test_pred)))
    #print(len(np.unique(Y_test)))

    #print(Y_test_pred)
    Y_test_pred_prob = np.exp(Y_test_pred) / np.sum(np.exp(Y_test_pred), axis=1, keepdims=True)
    #print(f"final dd : {len(np.unique(Y_test_pred_prob))}")
    auc_test = roc_auc_score(Y_test, Y_test_pred_prob, multi_class='ovr')
    print('AUC: ', auc_test)

    return model, auc_test, score[1], score[0]


#----------------------------------------------------------------
## Pre-Trained
#----------------------------------------------------------------
def PreIntraining_Model (model_pretrained,X_train,Y_train,X_test, Y_test, param_config):
    if X_train.shape[-1] == 1:
      X_train = np.repeat(X_train, 3, axis=-1)
    if X_test.shape[-1] == 1:
      X_test = np.repeat(X_test, 3, axis=-1)

    # Freeze all layers except for the
    #for layer in base_model.layers[:-13]:
    #    layer.trainable = False
    #model_pretrained.summary()

    if param_config.training_type == 1:
        model, auc, acc, loss = normal_train (model_pretrained,X_train,Y_train,X_test, Y_test,param_config)
    else:
      model, auc, acc, loss = image_data_generator (model_pretrained,X_train,Y_train,X_test, Y_test,param_config)

    return model, auc, acc, loss