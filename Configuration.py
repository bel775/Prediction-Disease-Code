from load_data import load_x_ray_data, load_Brain_Tumor_data, load_isic_data
from balance_Data_Using_Image_Gen import balance_Data
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras

class Configuration:
    def __init__(self,disease_type, input_shape, img_size, balance_type, 
                 model_type, num_filters1,num_filters2,num_filters3, 
                 kernel_size1, kernel_size2,kernel_size3, activation, 
                 training_type, batch_size, epochs, num_classes, optimization):
        self.disease_type = disease_type
        self.input_shape = input_shape
        self.img_size = img_size
        self.balance_type = balance_type
        self.model_type = model_type
        self.num_filters1 = num_filters1
        self.num_filters2 = num_filters2
        self.num_filters3 = num_filters3
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.activation = activation
        self.training_type = training_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.optimization = optimization

def Apply_AlexNet():
    while True:
        response = input("Want to apply AlexNet? (yes/no) [no]: ").strip().lower() or 'n'

        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def accept_digital_num(question, defult_num):
    while True:
        response = input(f"{question}[{defult_num}]: ") or defult_num

        if response.isdigit():
            return int(response)
        else:
            print(f"Invalid input. Please enter a digital number.")

def load_images(disease_type, img_size):
    if disease_type == '1':
        X, Y = load_x_ray_data(img_size)
    elif disease_type == '2':
        X, Y = load_Brain_Tumor_data(img_size)
    elif disease_type == '3':
        X, Y,X_Unbalanced,Y_Unbalanced = load_isic_data(img_size)
        Y_show_chart = np.concatenate((Y_Unbalanced, Y), axis=0)
        print_amount(Y_show_chart,disease_type)
        print("\nHave been using ImageDataGenerator automatically to handle imbalanced data.")
        X_balanced,Y_balanced = balance_Data(X_Unbalanced,Y_Unbalanced)

        X = np.concatenate((X_balanced, X), axis=0)
        Y = np.concatenate((Y_balanced, Y), axis=0)

    else: 
        X, Y = None, None
    return X,Y

def apply_balance_data(img_size,X,Y):
    response = input("Data balancing technique [1] OverSampler [2] UnderSampler [neither]: ") or 'n'

    if response == '1':
        oversample = RandomOverSampler()
        X, Y = oversample.fit_resample(X.reshape(len(X), -1), Y)
        X = X.reshape(-1, img_size, img_size, 1) 
        balance_type = 1
    elif response == '2':
        undersample = RandomUnderSampler()
        X, Y = undersample.fit_resample(X.reshape(len(X), -1), Y)
        X = X.reshape(-1, img_size, img_size, 1)
        balance_type = 2
    else: balance_type = 3
    
    return X, Y, balance_type


def show_chart(values, disease_type):
    rep = input("Show chart? (yes/no) [no]: ") or 'n'
    if rep in ['yes', 'y']:
        if disease_type == '1' or disease_type == '3':
            labels = ['Normal','BACTERIA','VIRUS']
            colors = ['blue', 'green', 'red']
            explode = [0.01, 0.01, 0.01]
        
        elif disease_type == '2':
            labels =  ['glioma','meningioma','notumor','pituitary']
            colors = ['blue', 'green', 'red', 'purple']
            explode = [0.01, 0.01, 0.01, 0.01]

        if disease_type == '3':
            labels = ['NEVUS','MELANOMA','Seborrheic Keratosis']

        plt.figure(figsize=(7, 5))
        plt.pie(values, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', textprops={"fontsize":15})

        # Añadir la leyenda
        plt.legend(labels, loc="upper left", bbox_to_anchor=(-0.2, 1.15)) #, bbox_to_anchor=(-0.1, 0.5), fontsize=12
        plt.title('Classification chart')
        plt.show()

        save_path = "Chart.png"
        plt.savefig(save_path)  # Save chart if interactive display fails
        print(f"Chart saved as {save_path}")

def print_amount(Y,disease_type):
    if disease_type == '1':
        normal = 0
        bacteria = 0
        virus = 0
        for Y_elem in Y:
            if Y_elem == 0: normal += 1
            elif Y_elem== 1 : bacteria += 1
            elif Y_elem == 2 : virus += 1

        print("\nNormal count : ",normal)
        print("Bacteria count : ",bacteria)
        print("Virus count : ",virus)

        values = [virus,bacteria,normal]
    
    elif disease_type == '2':
        glioma = 0
        meningioma = 0
        notumor = 0
        pituitary = 0

        for Y_elem in Y:
            if Y_elem == 0 : glioma += 1
            elif Y_elem == 1 : meningioma += 1
            elif Y_elem == 2 : notumor += 1
            elif Y_elem == 3 : pituitary += 1


        print("\nGlioma count : ",  glioma)
        print("Meningioma count : ",  meningioma)
        print("Notumor count : ",  notumor)
        print("Pituitary count : ",  pituitary)
        
        values = [glioma,meningioma,notumor,pituitary]
    
    elif disease_type == '3':
        nevus = 0
        melanoma = 0
        seborrheic_keratosis = 0

        for Y_elem in Y:
            if(Y_elem == 0): nevus += 1
            elif(Y_elem== 1): melanoma += 1
            elif(Y_elem == 2): seborrheic_keratosis += 1

        print("\nNevus count = ",nevus)
        print("Melanoma count = ",melanoma)
        print("Seborrheic Keratosis count = ",seborrheic_keratosis)

        values = [nevus,melanoma,seborrheic_keratosis]
    
    show_chart(values, disease_type)

activations = {
                "1": "relu",
                "2": layers.LeakyReLU(negative_slope=0.1),
                "3": "sigmoid",
                "4": "tanh",
                "5": "swish",
                "6": "gelu"
            }

optimizations = {
                "1": "adam",
                "2": "sgd",
                "3": "manual"
}

learning_rate = {
                "1": 1e-4,
                "2": 5e-5,
                "3": 1e-5
}

def optimization_question():
    while True:
        optimization_choice = input("Choose optimization [1] Adam [2] sgd [3] manual [Adam]: ") or "1"
        
        optimization = optimizations.get(optimization_choice)
        if optimization_choice not in ['1', '2', '3']:
            print("Invalid choice. Please enter a valid number (1, 2, or 3).")
        else:
            if optimization_choice in ['1', '2']:
                return optimization
            else:
                while True:
                    learning_rate_choice_str = input("Learning rate [1] 1e-4 [2] 5e-5 [3] 1e-5 [3]: ") or "3"
                    if learning_rate_choice_str not in ['1', '2', '3', '4']:
                        print("Invalid choice. Please enter a valid number (1, 2 or 3).")
                    else:
                        learning_rate_choice = learning_rate.get(learning_rate_choice_str)
                        op = keras.optimizers.Adam(learning_rate=learning_rate_choice)
                        return op


def training_type_question():
    while True:
        training_type = input("Choose training method [1] Normal Train [2] Data Generator: ")
        
        if training_type not in ['1', '2']:
            print("Invalid choice. Please enter a valid number (1 or 2).")
        else:
            return int(training_type)

def get_user_input():
    while True:
        disease_type = input("Type of disease you want to train [1] Chest X-Ray [2] Brain_Tumor [3] ISIC Melanoma: ")
        if disease_type not in {'1', '2', '3'}:
            print("Invalid choice. Please enter a valid number (1, 2 o 3).")
        else: 
            if disease_type == '1' or disease_type == '3':
                num_classes = 3
            elif disease_type == '2':
                num_classes = 4
            break

    apply_alexnet = Apply_AlexNet()

    if apply_alexnet:
        img_size = 224
    else:
        img_size_str = input("Input Image size [64]: ") or 64
        img_size= int(img_size_str)

    X,Y = load_images(disease_type, img_size)
    print_amount(Y,disease_type)
    X,Y,balance_type = apply_balance_data(img_size,X,Y)

    if apply_alexnet:

        model_type =  8
        num_filters1= 0
        num_filters2= 0
        num_filters3= 0
        kernel_size1= 0
        kernel_size2= 0
        kernel_size3= 0
        activation=""
        optimization = optimization_question()
        training_type = training_type_question()
        batch_size = accept_digital_num("Batch size ", "62")
        epochs = accept_digital_num("Number of epochs ", "32")

    else:

        while True:
            model_type = input("Choose model type [1] Functional API [2] Sequential API [3] SubClassing [4] Hybrid [5] ResNet [6] EfficientNet [7] VGG16: ")
            if model_type not in {'1', '2', '3', '4','5','6','7'}:
                print("Invalid choice. Please enter a valid number (1, 2, 3, 4, 5, 6 or 7).")
            else: break
        if model_type == '5' or model_type == '6' or model_type == '7':
            num_filters1 = 0
            num_filters2 = 0
            num_filters3 = 0
            kernel_size1 = 0
            kernel_size2 = 0
            kernel_size3 = 0
        else:
            num_filters1 = accept_digital_num("Number of filters for Layer 1 ", "16")
            num_filters2 = accept_digital_num("Number of filters for Layer 2 and 3 ", "64")
            num_filters3 = num_filters2
            kernel_size1 = accept_digital_num("Kernel size for Layer 1 ", "7")
            kernel_size2 = accept_digital_num("Kernel size for Layer 2 ", "5")
            kernel_size3 = accept_digital_num("Kernel size for Layer 3 ", "3")

        while True:
            activation_choice = input("Choose activation function [1] Relu [2] LeakyReLU [3] Sigmoid [4] Tanh [5] swish [6] gelu: ")
            
            activation = activations.get(activation_choice)
            if activation is None:
                print("Invalid choice. Please enter a valid number (1, 2, 3, 4, 5 or 6).")
            else:
                break

        optimization = optimization_question()

        training_type = training_type_question()

        batch_size = accept_digital_num("Batch size ", "62")
        epochs = accept_digital_num("Number of epochs ", "32")

    configuration = Configuration(
        disease_type = int(disease_type),
        img_size= img_size,
        balance_type = balance_type,
        input_shape= (img_size, img_size, 1),
        model_type = int(model_type),
        num_filters1= num_filters1,
        num_filters2= num_filters2,
        num_filters3= num_filters3,
        kernel_size1= kernel_size1,
        kernel_size2= kernel_size2,
        kernel_size3= kernel_size3,
        activation=activation,
        optimization = optimization,
        training_type = training_type,
        batch_size = batch_size,
        epochs = epochs,
        num_classes=num_classes
    )

    return X,Y,configuration

def print_Configuration(param_config):
    # Display the configured parameters
    print(f"\nConfigured CNN parameters:")
    print(f"Input Disease: {param_config.disease_type}")
    print(f"Image size: {param_config.img_size}")
    print(f"Input Shape: {param_config.input_shape}")
    print(f"Input model: {param_config.model_type}")
    print(f"Filters Layer 1: {param_config.num_filters1}")
    print(f"Filters Layer 2: {param_config.num_filters2}")
    print(f"Filters Layer 3: {param_config.num_filters3}")
    print(f"Kernel Size Layer 1: {param_config.kernel_size1}")
    print(f"Kernel Size Layer 2: {param_config.kernel_size2}")
    print(f"Kernel Size Layer 3: {param_config.kernel_size3}")
    print(f"Activation Function: {param_config.activation}")
    print(f"Number of Output Classes: {param_config.num_classes}")

    