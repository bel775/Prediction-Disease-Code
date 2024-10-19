
import sklearn
from models import get_sequential_api_model,get_funcional_api_model,CustomModel,get_alexnet_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from enviar_model import enviar_model_x_ray
from Configuration import get_user_input,print_Configuration

from tensorflow.keras.preprocessing import image
import numpy as np
import collections
from sklearn import metrics
import glob

img_size = 64

#Optimizer
#op = SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
op = 'adam' #'adam'

batch_size = 62
nb_classes = 3
epochs = 70

if __name__ == '__main__':
    X,Y,param_config = get_user_input()
    #print_Configuration(param_config)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=123)


    #model = cnn_model_functional_API(config)
    if param_config.model_type == 1:
        model = get_funcional_api_model(param_config)
    elif param_config.model_type == 2:
        model = get_sequential_api_model(param_config)
    elif param_config.model_type == 3:
        model = CustomModel(param_config)
        model.build(input_shape=(None, *param_config.input_shape))
    elif param_config.model_type == 4:
        model = get_alexnet_model(param_config)

    model.compile(loss='sparse_categorical_crossentropy',optimizer=op, metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=2)

    loss, acc = model.evaluate(X_test,Y_test,batch_size=batch_size)

    """ Resultado ROC"""
    y_pred = model.predict(X_test) 
    roc_auc = roc_auc_score(Y_test, y_pred, multi_class='ovr')
    print(f'AUC {roc_auc:.4f}')

    print('Predictions')
    y_pred_int = y_pred.argmax(axis=1)
    print(collections.Counter(y_pred_int))

    #print('Confusion matrix')
    #print(metrics.confusion_matrix(Y_test,y_pred_int))

    #print(metrics.classification_report(Y_test, y_pred_int, target_names=['Normal','BACTERIA','VIRUS']))

    #print(model.summary())
    enviar_model_x_ray(model,img_size,roc_auc,loss,acc,batch_size,epochs,op,activacion)
