# CACTI Code Library Documentation 
## folder configs
* folder __base__: model and basic dataset
* Each reconstruction algorithm configuration files
## folder cacti 
* datasets (data preprocessing)
* models (reconstruction algorithms)
* utils (universial function, such as PSNR SSIM Calculation)

## folder tools 
* train.py (Model Training)
* test_deeplearning.py (Testing end to end deep learning algorithms or deep unfolding algorithms on grayscale or colored simulation dataset)
* test_iterative.py (Testing iterative algorithms or plug and play algorithms on grayscale simulation dataset)
* test_color_iterative.py (Testing iterative algorithms or plug and play algorithms on colored simulation dataset)
* real_data (Testing real data)
* params_floats.py (Statistics of model parameters and FLOPs)
* video_gif (images to video and images to gif transfer)
* onnx_tensorrt (onnx, tensorrt model transfer and testing)

## folder test_datasets 
* mask (mask for different models)
* simulation (Testing results for 6 benchmark grayscale simulation dataset)
* middle_scale (Testing results for 6 benchmark colored simulation dataset)
* real_data (real data, compress ration from 10 to 50)



