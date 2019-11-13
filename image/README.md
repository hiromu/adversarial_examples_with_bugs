# Sample codes for image adversarial example

1. Download the target model from http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

2. Extract `inception_v3.ckpt` into `data/inception`

3. Clone the Costa Rica dataset from https://github.com/cvjena/costarica-moths intp `data/costarica-moths`

4. Download the images of the Costa Rica dataset by running `cd data/costarica-moths && bash download_images.sh`

5. Run `runner/1_preprocess/run.sh`

6. Choose one script from `runner/2_train_gan` corresponding to the size of perturbations and train a GAN model

7. (For the patch-based algorithm) choose one script from `runner/3_train_patch` corresponding to the size of perturbations and run as `bash runner/3_train_patch/run_32.sh input_image target_label`

8. (For the PGPE-based algorithm) choose one script from `runner/4_train_pgpe` corresponding to the size of perturbations and run as `bash runner/4_train_pgpe/run_32.sh input_image target_label`