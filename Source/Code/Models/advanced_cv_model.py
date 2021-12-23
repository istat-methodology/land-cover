from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.regularizers import l2
import sys
class AdvancedCVModel:
    
    @staticmethod
    #def build(neural_model, default_input_size, model_input_width, model_input_height, input_channels, model_output_n_classes, imagenet_weights, summary):
    def build(params):
        neural_model=params.neural_model
        default_input_size= params.default_input_size
        model_input_width= params.model_input_width
        model_input_height= params.model_input_height
        input_channels= params.eurosat_input_channels
        model_output_n_classes= params.model_output_n_classes
        imagenet_weights= params.imagenet_weights
        multiband_eurosat_data=params.multiband_eurosat_data
        summary= params.summary

        # Selecting whether to adopt transfer learning or not
        if imagenet_weights == True:
            pretrained_weights='imagenet' 
            if multiband_eurosat_data==True:
                print( "-----    Imeginet pretranied weights can not be used in multiband mode.    :) "  )
                sys.exit(0)
        else: 
            pretrained_weights=None
 
   
        if default_input_size == True: 
            if neural_model == 'resnet' or neural_model == 'inception': 
                model_input_height = 224
                model_input_width = 224
            elif neural_model == 'inceptionresnet' or neural_model == 'xception': 
                model_input_height = 299
                model_input_width = 299
                inputShape = (height, width, depth)
            elif neural_model == 'nasnet':
                model_input_height = 331
                model_input_width = 331

        if neural_model == 'xception':
            deep_network = Xception(weights=pretrained_weights, include_top=False, input_tensor=Input(shape=(model_input_width, model_input_height, input_channels)))
        elif neural_model == 'resnet':
            deep_network = ResNet50(weights=pretrained_weights, include_top=False, input_tensor=Input(shape=(model_input_width, model_input_height, input_channels)))
        elif neural_model.lower() == 'inception':
            deep_network = InceptionV3(weights=pretrained_weights, include_top=False, input_tensor=Input(shape=(model_input_width, model_input_height, input_channels)))
        elif neural_model.lower() == 'inceptionresnet':
            deep_network = InceptionResNetV2(weights=pretrained_weights, include_top=False, input_tensor=Input(shape=(model_input_width, model_input_height, input_channels)))
        elif neural_model == 'nasnet':
            deep_network = NASNetLarge(weights=pretrained_weights, include_top=False, input_tensor=Input(shape=(model_input_width, model_input_height, input_channels)))
        else: 
            print("Neural Model not supported: ", neural_model)
            return None    

        x = deep_network.output
        x = GlobalAveragePooling2D()(x)
        #x = Dense(128,activation='relu')(x)
        x = Dense(256,activation='relu')(x)
        #x = Dropout(0.2)(x)
        x = Dropout(0.5)(x)

        predictions = Dense(model_output_n_classes, kernel_regularizer = l2(0.005), activation='softmax')(x)

        model = Model(inputs=deep_network.input, outputs=predictions)

        if summary==True:
            model.summary()

        # return the constructed network architecture
        return model
