from load_data import load_x_ray_data, load_Brain_Tumor_data
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

class Configuration:
    def __init__(self,disease_type, input_shape, img_size, balance_type, model_type, num_filters1,num_filters2,num_filters3, kernel_size1, kernel_size2,kernel_size3, activation, num_classes):
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
        self.num_classes = num_classes

def Apply_AlexNet():
    while True:
        response = input("Want to apply AlexNet? (yes/no) [no]: ").strip().lower() or 'n'

        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def load_images(disease_type, img_size):
    if disease_type == '1':
        X, Y = load_x_ray_data(img_size)
    elif disease_type == '2':
        X, Y = load_Brain_Tumor_data(img_size)
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
        if disease_type == '1':
            labels = ['Normal','BACTERIA','VIRUS']
            colors = ['blue', 'green', 'red']
            explode = [0.01, 0.01, 0.01]
        
        elif disease_type == '2':
            labels =  ['glioma','meningioma','notumor','pituitary']
            colors = ['blue', 'green', 'red', 'purple']
            explode = [0.01, 0.01, 0.01, 0.01]

        plt.figure(figsize=(7, 5))
        plt.pie(values, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', textprops={"fontsize":15})

        # Añadir la leyenda
        plt.legend(labels, loc="upper left", bbox_to_anchor=(-0.2, 1.15)) #, bbox_to_anchor=(-0.1, 0.5), fontsize=12
        plt.title('Classification chart')
        plt.show()

def print_amount(Y,disease_type):
    if disease_type == '1':
        normal = 0
        bacteria = 0
        virus = 0
        for Y_elem in Y:
            if Y_elem == 0: normal += 1
            elif Y_elem== 1 : bacteria += 1
            elif Y_elem == 2 : virus += 1

        print("\nNormal count = ",normal)
        print("Bacteria count = ",bacteria)
        print("Virus count = ",virus)

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


        print("\nglioma: ",  glioma)
        print("meningioma: ",  meningioma)
        print("notumor: ",  notumor)
        print("pituitary: ",  pituitary)
        
        values = [glioma,meningioma,notumor,pituitary]
    
    show_chart(values, disease_type)



def get_user_input():
    while True:
        disease_type = input("Type of disease you want to train [1] X-Ray [2] Brain_Tumor: ")
        if disease_type not in {'1', '2'}:
            print("Invalid choice. Please enter a valid number (1 or 2).")
        else: 
            if disease_type == '1':
                num_classes = 3
            elif disease_type == '2':
                num_classes = 4
            break

    apply_alexnet = Apply_AlexNet()

    if apply_alexnet:
        img_size = 224
    else:
        img_size = input("Input Image size [64]: ") or 64

    X,Y = load_images(disease_type, img_size)
    print_amount(Y,disease_type)
    X,Y,balance_type = apply_balance_data(img_size,X,Y)

    if apply_alexnet:
        configuration = Configuration(
            disease_type = int(disease_type),
            img_size= int(img_size),
            balance_type = balance_type,
            input_shape= (img_size, img_size, 1),
            model_type = int("4"),
            num_filters1=int("0"),
            num_filters2=int("0"),
            num_filters3=int("0"),
            kernel_size1=int("0"),
            kernel_size2=int("0"),
            kernel_size3=int("0"),
            activation="",
            num_classes=int(num_classes)
        )

        return X,Y, configuration

    else:

        while True:
            model_type = input("Choose model type [1] Functional API [2] Sequential API [3] SubClassing: ")
            if model_type not in {'1', '2', '3'}:
                print("Invalid choice. Please enter a valid number (1, 2 or 3).")
            else: break

        num_filters1 = input("Number of filters for Layer 1 [16]: ") or 16
        num_filters2 = input("Number of filters for Layer 2 [32]: ") or 32
        num_filters3 = input("Number of filters for Layer 3 [64]: ") or 64
        kernel_size1 = input("Kernel size for Layer 1 [5]: ") or 5
        kernel_size2 = input("Kernel size for Layer 2 [5]: ") or 5
        kernel_size3 = input("Kernel size for Layer 3 [5]: ") or 5

        while True:
            activation_choice = input("Choose activation function [1] Relu [2] Sigmoid [3] Tanh: ")
            
            activations = {
                "1": "relu",
                "2": "sigmoid",
                "3": "tanh"
            }
            
            activation = activations.get(activation_choice)
            if activation is None:
                print("Invalid choice. Please enter a valid number (1, 2, or 3).")
            else:
                break

        configuration = Configuration(
            disease_type = int(disease_type),
            img_size= int(img_size),
            balance_type = balance_type,
            input_shape= (img_size, img_size, 1),
            model_type = int(model_type),
            num_filters1=int(num_filters1),
            num_filters2=int(num_filters2),
            num_filters3=int(num_filters3),
            kernel_size1=int(kernel_size1),
            kernel_size2=int(kernel_size2),
            kernel_size3=int(kernel_size3),
            activation=activation,
            num_classes=int(num_classes)
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

    