Due to the large size of the dataset (around 300MB), it is uploaded on Dropbox and can be downloaded via this link: 
https://www.dropbox.com/s/mzas2tkd80pubh7/data_covid5k.zip?dl=0


#COVID-XRay-5K DATASET

We prepared a dataset of around 5000 images, which can be downloaded from here: dataset_link

Two sources are used to create this dataset:

    Covid-Chestxray-Dataset, for COVID-19 X-ray samples
    ChexPert Dataset, for Non-COVID samples

COVID-19 samples from Covid-Chestxray-Dataset are extracted from a several publications, and it is important to verify all their labels. With the help of a board-certified radiologist, we went through X-ray images of COVID-19 samples, and only kept those which were selceted to have a clear sign of COVID-19 by our radiologist. We also only kept the posterior-anterior images.

For Non-COVID samples, we tried to uniformly sample images from ChexPert. More details on the dataset are provided in our paper.

Some of the sample images from our dataset are shown below. The images in the first row show COVID-19 cases, and the images in the remaining rows denote non-COVID cases.

samples

As the number of COVID-19 samples are much fewer than the number of Non-COVID samples, we used several data-augmentation techniques (as well as over-sampling) to increase the number of COVID-19 samples in training, to have a less imbalanced training set. Hopefully more cleanly labeled X-ray images from COVID-19 cases become available soon, so we do not have this imbalanced data issue.

For data augmentation, we have used the Augmentor library in Python.
