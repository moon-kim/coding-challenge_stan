# List files and pair with masks

import os
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


# input data directories for image and mask folders
dir_names = ["data/images/","data/masks/"]


# we need to extract the following features:
'''
feature  original_firstorder_10Percentile
feature  original_firstorder_90Percentile
feature  original_firstorder_Energy
feature  original_firstorder_Maximum
feature  original_firstorder_Mean
feature  original_firstorder_Median
feature  original_firstorder_RootMeanSquared
feature  original_firstorder_TotalEnergy
feature  original_glcm_Contrast
feature  original_glcm_DifferenceEntropy
'''
# feature key names of the reduced feature set
feature_names = ["original_firstorder_10Percentile",
                 "original_firstorder_90Percentile",
                 "original_firstorder_Energy",
                 "original_firstorder_Maximum",
                 "original_firstorder_Mean",
                 "original_firstorder_Median",
                 "original_firstorder_RootMeanSquared",
                 "original_firstorder_TotalEnergy",
                 "original_glcm_Contrast",
                 "original_glcm_DifferenceEntropy"
                ]

# first the images
image_filenames = []
for path in os.listdir(dir_names[0]):
    if(os.path.isfile(os.path.join(dir_names[0], path))):
        image_filenames.append(path) # we only want the filename, as they are the same for images and masks folder
        
# check, whether we have matching masks for each image in the masks folder

missing_mask = False

#just iterate through every filename in the image_filenames list and check whether it exists in the mask folder
for filename in image_filenames:
    if(not os.path.isfile(os.path.join(dir_names[1], filename))):
        missing_mask = True # when one is missing, set this bool to True
        
assert missing_mask == False, "images and mask files not matching!"

print("Data is okay!")


# these are the standard settings, may need some tunning with more information about the data we have
# would ask an expert at this point to provide better settings 

settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline



# Initialize feature extractor
radiomics_feature_extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

radiomics_feature_extractor.disableAllFeatures()

# these are the feature class names for the radiomics lib to enable feature extraction from these specific classes
feature_class_names = ["firstorder","glcm"]


# Enable all features in firstorder
for feature_class_name in feature_class_names:
    radiomics_feature_extractor.enableFeatureClassByName(feature_class_name)
    
class_names = ["bone","kidney","liver","muscle","spleen"]

test_set_x = []
test_set_y = []

#load data, extract features, extract reduced feature vector, get ground truth class 

for filename in image_filenames:
    absolute_filenames = [dir_names[i]+filename for i in range(2)] # absolute filenames for images and masks
    
    raw_feature_vector = radiomics_feature_extractor.execute(absolute_filenames[0],absolute_filenames[1])    
    feature_vector = []
    #filter the feature vector so we get a dict with only features and drop the extra info
    for j in range(len(feature_names)):
        for feature_key in raw_feature_vector.keys():
            if(feature_key == feature_names[j]): 
                feature_vector.append(float(raw_feature_vector[feature_key]))
    test_set_x.append(feature_vector)
    
    #get the class from the filename
    for j,class_name in enumerate(class_names):
        if(class_name in absolute_filenames[0]): 
            test_set_y.append(j)

            
            
test_set_x = np.array(test_set_x)
test_set_y = np.array(test_set_y)

# load model and calculate score
estimator = pickle.load(open("final_estimator.h5", 'rb'))
test_accuracy = estimator.score(test_set_x,test_set_y)
print("The test accuracy is :",test_accuracy*100, "%")
