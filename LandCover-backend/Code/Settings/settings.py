'''
Created on 01/01/2020
Modified on 17/01/2020

@author: Fabrizio De Fausti, Francesco Pugliese
'''

import configparser
import pdb

class SetParameters:
    
    def __init__(self, conf_file_path, conf_file_name, OS):
        # Class Initialization (constructor) 
        self.conf_file_path = conf_file_path
        self.conf_file_name = conf_file_name
        
        # System
        self.lc_mode = 0                                                     # 0: classify, 1: preprocess, 2: training, 3: testing
        self.gpu = False
        self.gpu_id = '0'
        self.system_desc = 'Laptop'
        self.gpu_desc = None

        # Dataset
        #self.eurosat_data = True
        self.eurosat_dataset_path = ""
        self.eurosat_dataset_training_set_path = ""
        self.eurosat_dataset_validation_set_path = ""
        self.eurosat_input_size = 64
        self.eurosat_input_channels = 3
        #self.eurosat_classification_type = 'EuroSAT'
        
        self.multiband_eurosat_data = False
        #self.eurosat_multiband_dataset_path = ""
        #self.nir = False
        #self.rgb_saturation_level = 2000          
        #self.other_bands_saturation_level = 5000          
        #self.eurosat_multiband_input_channels = 13
        #self.eurosat_multiband_output_size = 10
        #self.eurosat_multiband_classification_type = 'EuroSAT Multiband'

        # Preprocessing
        self.dataAugmentation = False
        self.saveAugmentedData = False
        self.augmented_images_path = ''
        self.valid_set_perc = 3                                          # Validation set percentage with respect to the Training Set
        self.test_set_perc = 3                                           # Test set percentage with respect to the entire Data Set
        #self.normalize_x = False                                         # Normalize input between 0 and 1, time-consuming for bigger datasets
        #self.normalize_y = False                                         # Normalize input between 0 and 1, time-consuming for bigger datasets
        self.limit_per_class = None
        self.img_to_array = True
        self.rescale = True                                              # True: adapt the data size to required input size, False: adapt input layer to the data size                                                                                                                                                                                                    
        
        # Model
        self.imagenet_weights = False
        self.neural_model = 'Inception'
        self.models_path = '../SavedModels'
        self.model_file = 'eurosat_land_cover_model.cnn'
        self.default_input_size = False
        self.model_input_width = 139
        self.model_input_height = 139
        self.summary = True
        self.model_output_n_classes = 10
        self.exclude_classes = None
        
  
        # Training
        self.train_tag = ""
        self.epochs_number = 2
        self.learning_rate = 0.0001                                      # best learning rate = 0.0015 at moment, 0.001 on mcover
        self.train_batch_size = 32
        self.training_algorithm = 'sgd'
        self.early_stopping = True
        self.save_best_model = True
        self.output_training_path = 'Output_Training'
        self.output_classes='output_classes.csv'
        # Testing
        #self.model_testing_file = ''
        self.model_testing_dir = self.output_training_path
        self.test_batch_size = 64

        # Classification
        self.stride = 64
        self.rotate_tiles = False
        self.random_rotations = False
        self.quantization = False
        self.n_samples = 3        
        self.parallelize = False
        self.interface = True
        self.test_batch_size = 32
        self.input_image = 'Test_examples/Eurosat_example_01.jpg'
        self.input_path = ''
        self.multiple_inputs = False
        self.save_tmp_tile = True
        self.classes_color='classes_color.csv'

	#Postprocessing


        # Output
        self.output_path = '../Output'
        self.log_path = '../Log'
        self.save_tiles = False
        self.charts_path = '../Output/Charts' 
        self.csv_path = '../Output/Csv'
        self.tiles_path = '../Output/Tiles'
        self.print_confusion_matrix = True
        self.plot_history = False
        self.plot_classification = False
        self.plot_lc_maps = False
        self.save_lc_statistics_pieplot = False
        self.lc_statistics_pieplot_file = 'lc_statistics_pieplot.jpg'
        self.plots_show = False
        self.save_execution_times = False
        self.times_files = 'lc_times.csv'
        self.pause_time = 5
        
        # Others
        self.OS = OS
        
        # Global Constants
        # Alert Sound
        self.sound_freq = 1000                                           # Set Frequency in Hertz
        self.sound_dur = 3000                                            # Set Duration in ms, 1000 ms == 1 second
        self.times_header = ["Preprocessing Time", "Postprocessing Time", "Gpu Time", "Overall Time without Gpu", "Overall Time"]
        self.model_prefix = ''
        
    def read_config_file(self):
        # Read the Configuration File
        config = configparser.ConfigParser()
        config.read(self.conf_file_path+'/'+self.conf_file_name)
        config.sections()

        # System
        self.lc_mode = config.getint('System', 'lc_mode')                                                     # 0: classify, 1: preprocess, 2: training, 3: testing
        self.gpu = config.getboolean('System', 'gpu')
        self.gpu_id = config.get('System','gpu_id')
        self.system_desc = config.get('System', 'system_desc')
        self.gpu_desc = config.get('System','gpu_desc')
        if self.gpu_desc == 'None': 
            self.gpu_desc = None

        # Dataset
        #self.eurosat_data = config.getboolean('Dataset', 'eurosat_data')

        if self.OS == "Linux":
            self.eurosat_dataset_path = config.get('Dataset', 'eurosat_dataset_path_linux')
            self.eurosat_dataset_training_set_path = config.get('Dataset', 'eurosat_dataset_training_set_path_linux')
            self.eurosat_dataset_validation_set_path = config.get('Dataset', 'eurosat_dataset_validation_set_path_linux')
        elif self.OS == "Windows": 
            self.eurosat_dataset_path = config.get('Dataset', 'eurosat_dataset_path_win')
            self.eurosat_dataset_training_set_path = config.get('Dataset', 'eurosat_dataset_training_set_path_win')
            self.eurosat_dataset_validation_set_path = config.get('Dataset', 'eurosat_dataset_validation_set_path_win')

        self.eurosat_input_size = config.getint('Dataset', 'eurosat_input_size')
        self.eurosat_input_channels = config.getint('Dataset', 'eurosat_input_channels')

        #self.eurosat_classification_type = config.get('Dataset', 'eurosat_classification_type')
        self.multiband_eurosat_data = config.getboolean('Dataset', 'multiband_eurosat_data')
        #self.eurosat_multiband_dataset_path = config.get('Dataset', 'eurosat_multiband_dataset_path')
        #self.nir = config.getboolean('Dataset', 'nir')
        #self.rgb_saturation_level = config.getint('Dataset', 'rgb_saturation_level')          
        #self.other_bands_saturation_level = config.getint('Dataset', 'other_bands_saturation_level')          
        #self.eurosat_multiband_input_channels = config.getint('Dataset', 'eurosat_multiband_input_channels')
        #self.eurosat_multiband_output_size = config.getint('Dataset', 'eurosat_multiband_output_size')
        #self.eurosat_multiband_classification_type = config.get('Dataset', 'eurosat_multiband_classification_type')
        #self.rescale = config.getboolean('Dataset', 'rescale')
        
        # Preprocessing
        self.dataAugmentation = config.getboolean('Preprocessing', 'dataAugmentation')
        self.saveAugmentedData = config.getboolean('Preprocessing', 'saveAugmentedData')
        
        if self.OS == "Linux":
            self.augmented_images_path = config.get('Preprocessing', 'augmented_images_path_linux')
        elif self.OS == "Windows": 
            self.augmented_images_path = config.get('Preprocessing', 'augmented_images_path_win')
        
        self.valid_set_perc = config.getint('Preprocessing', 'valid_set_perc')                                           # Validation set percentage with respect to the Training Set
        self.test_set_perc = config.getint('Preprocessing', 'test_set_perc')
        #self.normalize_x = config.getboolean('Preprocessing', 'normalize_x')
        #self.normalize_y = config.getboolean('Preprocessing', 'normalize_y')
        self.limit_per_class = config.get('Preprocessing', 'limit_per_class')
        try: 
            self.limit_per_class = int(self.limit_per_class)
        except ValueError: 
            self.limit_per_class = None
        self.img_to_array = config.getboolean('Preprocessing', 'img_to_array')

        # Model
        self.imagenet_weights = config.getboolean('Model', 'imagenet_weights')
        self.neural_model = config.get('Model', 'neural_model')
        
        if self.OS == "Linux":
            self.models_path = config.get('Model', 'models_path_linux')
        elif self.OS == "Windows": 
            self.models_path = config.get('Model', 'models_path_win')
        
        self.model_file = config.get('Model', 'model_file')
        self.default_input_size = config.getboolean('Model', 'default_input_size')
        self.model_input_width = config.getint('Model', 'model_input_width')
        self.model_input_height = config.getint('Model', 'model_input_height')
        self.summary = config.getboolean('Model', 'summary')
        self.model_output_n_classes = config.getint('Model', 'model_output_n_classes')
        self.exclude_classes = config.get('Model', 'exclude_classes')
        if self.exclude_classes == 'None': 
            self.exclude_classes = None
            
        # Training
        self.epochs_number = config.getint('Training', 'epochs_number')
        self.learning_rate = config.getfloat('Training', 'learning_rate')
        self.train_batch_size = config.getint('Training', 'train_batch_size')
        self.training_algorithm = config.get('Training', 'training_algorithm')
        self.early_stopping = config.getboolean('Training', 'early_stopping')
        self.save_best_model = config.getboolean('Training', 'save_best_model')

        # Testing
        self.model_testing_dir = config.get('Testing', 'model_testing_dir')
        self.test_batch_size = config.getint('Testing', 'test_batch_size')

        # Classification
        self.stride = config.getint('Classification', 'stride')
        self.rotate_tiles = config.getboolean('Classification', 'rotate_tiles')
        self.random_rotations = config.getboolean('Classification', 'random_rotations')
        self.quantization = config.getboolean('Classification', 'quantization')
        self.n_samples = config.getint('Classification', 'n_samples')
        self.interface = config.getboolean('Classification', 'interface')
        self.test_batch_size = config.getint('Classification', 'test_batch_size')
        self.input_image = config.get('Classification', 'input_image')
        self.multiple_inputs = config.getboolean('Classification', 'multiple_inputs')
        self.input_path = config.get('Classification', 'input_path')
        self.save_tmp_tile = config.getboolean('Classification', 'save_tmp_tile')
        #self.model_classify_file = config.get('Classification', 'model_classify_file')
        self.model_classification_dir = config.get('Classification', 'model_classification_dir')
        self.dim_model_input_image = config.getint('Classification', 'dim_model_input_image')
        self.mapping_eurosat_lucas = config.get('Classification', 'mapping_eurosat_lucas')
        self.tag = config.get('Classification', 'tag')
        self.other_image_type = config.get('Classification', 'other_image_type')
	
	#Postprocessing
        #self.input_classification_matrix_file = config.get('Postprocessing', 'input_classification_matrix_file') 

        # Output
        self.output_path = config.get('Output', 'output_path')
        self.log_path = config.get('Output', 'log_path')
        self.save_tiles = config.getboolean('Output', 'save_tiles')
        self.charts_path = config.get('Output', 'charts_path') 
        self.csv_path = config.get('Output', 'csv_path')
        self.tiles_path = config.get('Output', 'tiles_path')
        self.print_confusion_matrix = config.getboolean('Output', 'print_confusion_matrix')
        self.plot_history = config.getboolean('Output', 'plot_history')
        self.plot_classification = config.getboolean('Output', 'plot_classification')
        self.plot_lc_maps = config.getboolean('Output', 'plot_lc_maps')
        self.save_lc_statistics_pieplot = config.getboolean('Output', 'save_lc_statistics_pieplot')
        self.lc_statistics_pieplot_file = config.get('Output', 'lc_statistics_pieplot_file') 
        self.plots_show = config.getboolean('Output', 'plots_show')
        self.save_execution_times = config.getboolean('Output', 'save_execution_times')
        self.times_files = config.get('Output', 'times_files')
        self.pause_time = config.getint('Output', 'pause_time')
		
        # Global Constants
        self.model_prefix = self.gpu_desc.lower()+'_'+self.OS.lower()+'_'+str(self.model_input_height)+'_height_'+str(self.epochs_number)+'_epochs_model_'+self.neural_model.lower()+'_'
        self.train_output_header = ' '.join([x.capitalize() for x in self.model_prefix.split('_')])
        self.test_output_header = ' '.join([x.capitalize() for x in self.model_testing_dir.split('.')[0].replace('linux', self.OS).replace('windows', self.OS).split('_')])
        return self		
