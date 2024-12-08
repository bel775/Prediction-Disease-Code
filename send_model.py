import requests
import base64
import json
from SecuritySys import cargar_clave_publica, cifrar_datos

#from tensorflow.keras.models import model_from_json
import numpy as np

public_key = cargar_clave_publica("public_key.pem")
api_key = "6609c948-c626-4ab4-8e09-f98420bdf2fd"

def send_model(model,param_config,roc_auc,loss,acc):

    cifrar_disease_type = cifrar_datos(str(param_config.disease_type), public_key)
    cifrar_img_size = cifrar_datos(str(param_config.img_size), public_key)
    cifrar_balance_type = cifrar_datos(str(param_config.balance_type), public_key)
    cifrar_model_type = cifrar_datos(str(param_config.model_type), public_key)
    cifrar_activation = cifrar_datos(param_config.activation, public_key)
    cifrar_training_type = cifrar_datos(str(param_config.training_type), public_key)
    cifrar_batch_size = cifrar_datos(str(param_config.batch_size), public_key)
    cifrar_epochs = cifrar_datos(str(param_config.epochs), public_key)
    cifrar_optimization = cifrar_datos(str(param_config.optimization), public_key)

    cifrar_roc_auc = cifrar_datos(str(roc_auc), public_key)
    cifrar_loss = cifrar_datos(str(loss), public_key)
    cifrar_acc = cifrar_datos(str(acc), public_key)

    # Convert model weights
    #model_file_path = "model.keras"
    #model.save(model_file_path)

    weights = model.get_weights()
    weights_as_lists = [w.tolist() for w in weights]
    model_json = model.to_json()
    weights_base64 = base64.b64encode(json.dumps(weights_as_lists).encode()).decode('utf-8')
    model_base64 = base64.b64encode(model_json.encode('utf-8')).decode('utf-8')




    # Base64 encode encrypted data
    cifrar_disease_type_base64 = base64.b64encode(cifrar_disease_type).decode('utf-8')
    cifrar_img_size_base64 = base64.b64encode(cifrar_img_size).decode('utf-8')
    cifrar_balance_type_base64 = base64.b64encode(cifrar_balance_type).decode('utf-8')
    cifrar_model_type_base64 = base64.b64encode(cifrar_model_type).decode('utf-8')
    cifrar_activation_base64 = base64.b64encode(cifrar_activation).decode('utf-8')
    cifrar_training_type_base64 = base64.b64encode(cifrar_training_type).decode('utf-8')
    cifrar_batch_size_base64 = base64.b64encode(cifrar_batch_size).decode('utf-8')
    cifrar_epochs_base64 = base64.b64encode(cifrar_epochs).decode('utf-8')
    cifrar_optimization_base64 = base64.b64encode(cifrar_optimization).decode('utf-8')
    cifrar_roc_auc_base64 = base64.b64encode(cifrar_roc_auc).decode('utf-8')
    cifrar_loss_base64 = base64.b64encode(cifrar_loss).decode('utf-8')
    cifrar_acc_base64 = base64.b64encode(cifrar_acc).decode('utf-8')




    model_data = {
        "model" : model_base64,
        "weights": weights_base64,
        "roc_auc": cifrar_roc_auc_base64,
        "loss": cifrar_loss_base64,
        "accuracy": cifrar_acc_base64,
        "disease_type": cifrar_disease_type_base64,
        "img_size": cifrar_img_size_base64,
        "balance_type": cifrar_balance_type_base64,
        "model_type": cifrar_model_type_base64,
        "activation": cifrar_activation_base64,
        "training_type": cifrar_training_type_base64,
        "batch_size": cifrar_batch_size_base64,
        "epochs": cifrar_epochs_base64,
        "optimization": cifrar_optimization_base64
    }

    headers = {
        "Content-Type": "application/json",
        "API-Key": api_key 
    }

    url = "http://localhost:2024/get_new_model"

    try:

        response = requests.post(url, json=model_data, headers=headers)
        if response.status_code == 200:
            print("Model and parameters sent successfully:", response.text)
        else:
            print("Failed to send model and parameters:", response.status_code, response.text)
    except requests.RequestException as e:
        print("Request failed:", e)

    """with open(model_file_path, "rb") as model_file:
        files = {
            "model_file": model_file,  # The .keras file
        }

    try:
        respuesta = requests.post(url,files=files, json=model_data, headers=headers, verify=False)  # verify='certificado_autofirmado.pem'
        if respuesta.status_code == 200:
            print("successful request:")
            print(respuesta.text)
        else:
            print(f"Error in the request: {respuesta.status_code}")
            print(respuesta.text)
    except requests.RequestException as e:
        print(f"Request failed: {e}")"""

    #model_2 = model_from_json(model_json)
    #model_2.set_weights(weights)
#
    #new_pred = model_2.predict(X_test)
    #np.testing.assert_allclose(y_pred, new_pred, rtol=1e-6, atol=1e-6)