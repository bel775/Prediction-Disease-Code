
from Configuration import get_user_input,print_Configuration
from cv_10 import cnn_cross_validation
from send_model import send_model
import matplotlib.pyplot as plt

import numpy as np
import collections
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#Optimizer
#op = SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

def print_chart(auc_array,acc_array,loss_array):
    x = np.arange(1, len(auc_array) + 1) 

    plt.figure(figsize=(10, 6)) 

    plt.plot(x, auc_array, marker='o', linestyle='-', color='b', label='auc')
    plt.plot(x, acc_array, marker='s', linestyle='--', color='r', label='accurcy')
    plt.plot(x, loss_array, marker='p', linestyle=':', color='g', label='loss')

    plt.xlabel('Number of CV-10')
    plt.title('Values of AUC, accuracy, and loss for each CV-10') 
    plt.legend()  

    plt.grid(True) 
    plt.tight_layout()  

    plt.show() 

def print_result(auc_array, acc_array, loss_array):
    # Initialize variables for averages
    medio_auc = 0
    medio_acc = 0
    medio_loss = 0
    
    # Print header for the table
    print(f'{"Loss":<10} {"Accuracy":<10} {"AUC":<10}')
    print('-' * 30)
    
    # Loop through the arrays and print values in table format
    for auc, acc, loss in zip(auc_array, acc_array, loss_array):
        medio_auc += auc
        medio_acc += acc
        medio_loss += loss
        print(f'{loss:<10.4f} {acc:<10.4f} {auc:<10.4f}')
    
    # Print separator
    print('-' * 30)
    
    # Print averages
    print(f'{(medio_loss / len(loss_array)):<10.4f} {(medio_acc / len(acc_array)):<10.4f} {(medio_auc / len(auc_array)):<10.4f}')

    print_chart(auc_array,acc_array,loss_array)


if __name__ == '__main__':
    X,Y,param_config = get_user_input()
    #print_Configuration(param_config)

    X, X_final_test, Y, Y_final_test = train_test_split(X, Y, test_size=0.20, random_state=123)

    best_model,array_auc, array_acc, array_loss = cnn_cross_validation(X,Y, param_config)
    print_result(array_auc, array_acc, array_loss)

    if param_config.model_type == 5 or param_config.model_type == 6 or param_config.model_type == 7:
        if X_final_test.shape[-1] == 1:
            X_final_test = np.repeat(X_final_test, 3, axis=-1)

    #Final Evaluation
    loss, acc = best_model.evaluate(X_final_test, Y_final_test, batch_size=param_config.batch_size)
    print(f'final model loss: {loss:.4f} best model acc: {acc:.4f}')

    y_pred = best_model.predict(X_final_test)
    y_pred_normalized = y_pred / np.sum(y_pred, axis=1, keepdims=True)

    roc_auc = roc_auc_score(Y_final_test, y_pred_normalized, multi_class='ovr')
    print(f'Final AUC {roc_auc:.4f}')

    print('Predictions')
    y_pred_int = y_pred_normalized.argmax(axis=1)
    print(collections.Counter(y_pred_int),'\n')

    print('Metrics')
    if param_config.disease_type == 1:
        print(metrics.classification_report(Y_final_test, y_pred_int, target_names=['Normal','BACTERIA','VIRUS']))
    elif param_config.disease_type == 2:
        print(metrics.classification_report(Y_final_test, y_pred_int, target_names=['Glioma','Meningioma','Notumor','Pituitary']))
    elif param_config.disease_type == 3:
        print(metrics.classification_report(Y_final_test, y_pred_int, target_names=['Nevus','Melanoma','Seborrheic keratosis']))

    print('Confusion matrix')
    if param_config.disease_type == 1:
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(Y_final_test,y_pred_int), display_labels=['Normal','BACTERIA','VIRUS']).plot()
    elif param_config.disease_type == 2:
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(Y_final_test, y_pred_int), display_labels=['Glioma','Meningioma','Notumor','Pituitary']).plot()
    elif param_config.disease_type == 3:
        metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(Y_final_test, y_pred_int), display_labels=['Nevus','Melanoma','Seborrheic keratosis']).plot()
    plt.show()


    #best_model.summary()
    while True:
        response = input("Want to send the model to the siteweb? (yes/no) [no]: ").strip().lower() or 'n'

        if response in ['yes', 'y']:
            send_model(best_model,param_config,roc_auc,loss,acc)
            break
        elif response in ['no', 'n']:
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
