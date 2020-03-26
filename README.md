# Traffic Sign Classifier (GTSRB)

This is an implementation of a traffic sign classifier using Artificial Neural Networks (ANN). It is obviously better to use Convolutional Neural Networks (CNN) (I achieved 99.54% in test set using Darknet-53), but this code aims to show that ANN are slower and have lower metric scores than CNN in a computer vision problem.

This neural network achieved a test set F1 score of 91.55%, using the hyper-parameters in the file parameters.py

# How to run

In order to use this repository, you must download the dataset first:

Download training set:

```bash
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
```

Download testing set:

```bash
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
```

Uncompress these files in the folder (create it) dataset/.

Your dataset/ folder should look like this:

- dataset/GTSRB/Final_Training/Images/

- dataset/GTSRB/Final_Test/Images/

Now, download the testing set annotations inside the folder dataset/GTSRB/Final_Test/Images/

Download testing set annotations:

```bash
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
```

If you want to, edit the hyper-parameters in parameters.py

Now you are ready to run the code:

```bash
python3 train.py
```
