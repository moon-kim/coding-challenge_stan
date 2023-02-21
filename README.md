# Translational Image-Guided Oncology: Coding Challenge

Please checkout the notebooks 1_data_checkout and 2_radiometric_feature_journey to get an idea about my approaches and ideas. 
As I wrote in the first notebook, I made a little mistake and jumped into the challange as a image-based classfication problem without ay prior knowledge about the meaning. I think I got some good ideas there, how to solve it, when I wouldn't have any additional information, besides the raw image data. 
As there is lot of information from the radiometric feature extraction, I built my estimator based on them, which can be seen in the second notebook.

Use the requirements.txt to create an env with the needed libraries.

To run the estimator on new data, please open the predict.py file and change the image and mask folder names with your test data at the top of the file. You can run it and it will show you the test accuracy against the ground truth of the image-filename classes. 

The environment was created with Python 3.10 and virtualenv.

## Important

When you rerun the second notebook to generate a new estimator, please be sure, that the selected features are the same, as the predict.py features !!! When they differ, we extract different features, as the classifier was trained on! 
To change that, copy the selected feature names into the "feature_names" array from predict.py to addapt


