from models import get_sequential_api_model,get_functional_api_model,CustomModel,get_hybrid_model,get_alexnet_model,get_preTrained_ResNet,get_preTrained_EfficientNet, get_preTrained_VGG16
from training_methods import normal_train, image_data_generator, PreIntraining_Model
from sklearn.model_selection import KFold
from tensorflow import keras
import numpy as np

def get_model(config):
    if config.model_type == 1:
        model = get_functional_api_model(config)
    elif config.model_type == 2:
        model = get_sequential_api_model(config)
    elif config.model_type == 3:
        model = CustomModel(config)
        model.build(input_shape=(None, *config.input_shape))
    elif config.model_type == 4:
        model = get_hybrid_model(config)
    elif config.model_type == 5:
        model = get_preTrained_ResNet(config)
    elif config.model_type == 6:
        model = get_preTrained_EfficientNet(config)
    elif config.model_type == 7:
        model = get_preTrained_VGG16(config)
    elif config.model_type == 8:
        model = get_alexnet_model(config)
    return model

def cnn_cross_validation(X,Y, param_config):
    kf = KFold(n_splits=10,shuffle=True, random_state=123)
    auc_aux, acc_aux, loss_aux = [],[],[]

    best_auc = 0
    best_model = get_model(param_config)

    for train_index, test_index in kf.split(X):
        keras.backend.clear_session()
        model = get_model(param_config)

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        optimizer = param_config.optimization
        if optimizer not in ['adam','sgd']:
            optimizer = param_config.optimization.__class__()
        model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

        if param_config.model_type == 5 or param_config.model_type == 6 or param_config.model_type == 7:
            model_aux, auc, acc, loss = PreIntraining_Model (model,X_train,Y_train,X_test, Y_test, param_config)
        else:
            """if param_config.model_type == 7:
                if X_train.shape[-1] == 1:
                    X_train = np.repeat(X_train, 3, axis=-1)
                if X_test.shape[-1] == 1:
                    X_test = np.repeat(X_test, 3, axis=-1)"""

            if param_config.training_type == 1:
                model_aux, auc, acc, loss = normal_train (model,X_train,Y_train,X_test, Y_test,param_config)
            elif param_config.training_type == 2:
                model_aux, auc, acc, loss = image_data_generator (model,X_train,Y_train,X_test, Y_test, param_config)

        if(auc > best_auc):
            best_model = model_aux
            best_auc = auc



        auc_aux.append(auc)
        acc_aux.append(acc)
        loss_aux.append(loss)
    return best_model,auc_aux,acc_aux,loss_aux