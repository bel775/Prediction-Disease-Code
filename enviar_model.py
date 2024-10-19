import requests
import base64
import json
from Seguridad import cargar_clave_publica, cifrar_datos

from tensorflow.keras.models import model_from_json
import numpy as np

public_key = cargar_clave_publica("public_key.pem")
api_key = "6609c948-c626-4ab4-8e09-f98420bdf2fd"

def enviar_model_x_ray(model,img_size,roc_auc,loss,acc,batch_size,epochs,op,activacion):
    model_util = "Funcional API"

    cifrar_img_size = cifrar_datos(str(img_size),public_key)
    cifrar_roc_auc = cifrar_datos(str(roc_auc),public_key)
    cifrar_loss = cifrar_datos(str(loss),public_key)
    cifrar_acc = cifrar_datos(str(acc),public_key)
    cifrar_batch_size = cifrar_datos(str(batch_size),public_key)
    cifrar_epochs = cifrar_datos(str(epochs),public_key)
    cifrar_op = cifrar_datos(op,public_key)
    cifrar_activacion = cifrar_datos(activacion,public_key)
    cifrar_model_util = cifrar_datos(model_util,public_key)


    
    weights = model.get_weights()
    weights_as_lists = [w.tolist() for w in weights]
    model_json = model.to_json()
    weights_base64 = base64.b64encode(json.dumps(weights_as_lists).encode()).decode('utf-8')


    img_size_cifrado_base64 = base64.b64encode(cifrar_img_size).decode('utf-8')
    cifrar_roc_auc_base64 = base64.b64encode(cifrar_roc_auc).decode('utf-8')
    cifrar_loss_base64 = base64.b64encode(cifrar_loss).decode('utf-8')
    cifrar_acc_base64 = base64.b64encode(cifrar_acc).decode('utf-8')
    cifrar_batch_size_base64 = base64.b64encode(cifrar_batch_size).decode('utf-8')
    cifrar_epochs_base64 = base64.b64encode(cifrar_epochs).decode('utf-8')
    cifrar_op_base64 = base64.b64encode(cifrar_op).decode('utf-8')
    cifrar_activacion_base64 = base64.b64encode(cifrar_activacion).decode('utf-8')
    cifrar_model_util_base64 = base64.b64encode(cifrar_model_util).decode('utf-8')



    datos = {
        "model": model_json,
        "weights": weights_base64,
        "img_size": img_size_cifrado_base64,
        "roc_auc" : cifrar_roc_auc_base64,
        "models_utl" : cifrar_model_util_base64,
        "batch_size" : cifrar_batch_size_base64,
        "epochs" : cifrar_epochs_base64,
        "accuracy" : cifrar_acc_base64,
        "loss" : cifrar_loss_base64,
        "op" : cifrar_op_base64,
        "activacion" : cifrar_activacion_base64
    }

    headers = {
        "Content-Type": "application/json",
        "API-Key": api_key  # Reemplaza "Api-Key" con el nombre real del encabezado utilizado por la API
    }

    url = "http://localhost:2024/model_x_ray"
    respuesta = requests.post(url, json=datos, headers=headers,verify=False) #verify='certificado_autofirmado.pem'

    if respuesta.status_code == 200:
        print("Solicitud exitosa:")
        print(respuesta.text)
    else:
        print(f"Error en la solicitud: {respuesta.status_code}")
        print(respuesta.text)




    #model_2 = model_from_json(model_json)
    #model_2.set_weights(weights)
#
    #new_pred = model_2.predict(X_test)
    #np.testing.assert_allclose(y_pred, new_pred, rtol=1e-6, atol=1e-6)