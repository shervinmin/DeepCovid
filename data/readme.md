## Download The Data
Due to the relatively large size of the dataset (around 300MB), it is uploaded on Dropbox and can be downloaded via this link: 
https://www.dropbox.com/s/mzas2tkd80pubh7/data_covid5k.zip?dl=0


## COVID-XRay-5K Dataset Description

We prepared a dataset of around 5000 images, which can be downloaded from here: [dataset_link](https://www.dropbox.com/s/mzas2tkd80pubh7/data_covid5k.zip?dl=0)

Two sources are used to create this dataset:
* [Covid-Chestxray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset), for COVID-19 X-ray samples
* [ChexPert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), for Non-COVID samples

COVID-19 samples from Covid-Chestxray-Dataset are extracted from a several publications, and it is important to verify all their labels. With the help of a board-certified radiologist, we went through X-ray images of COVID-19 samples, and only kept those which were selceted to have a clear sign of COVID-19 by our radiologist. We also only kept the posterior-anterior images. 

For Non-COVID samples, we tried to uniformly sample images from ChexPert. More details on the dataset are provided in our [paper](https://arxiv.org/pdf/2004.09363.pdf).

Some of the sample images from our dataset are shown below. The images in the first row show COVID-19 cases, and the images in the remaining rows denote non-COVID cases.

![samples](https://github.com/shervinmin/DeepCovid/blob/master/results/covid5k_samples.png)

Out of all COVID-19 X-ray images in **Covid-Chestxray-Dataset** (more than 100 images), a total of 71 images are verified by our board-certified radiologist to have a clear sign of COVID-19, and are used in our dataset. Out of these images, 31 are used for training and 40 for test images (due to some consideration w.r.t. maximum confidence interval for sensitivity rate). 

As the number of COVID-19 samples are much fewer than the number of Non-COVID samples, we used several data-augmentation techniques (as well as over-sampling) to increase the number of COVID-19 samples in training, to have a less imbalanced training set. Hopefully more cleanly labeled X-ray images from COVID-19 cases become available soon, so we do not have this imbalanced data issue.

The number of samples from each class (COVID-19, and Non-COVID) in our dataset is shown below:

| Split         | COVID-19      | Non-COVID  |
| ------------- |:-------------:| -----:|
| Training Set  | 31  (496 after augmentation) | 2000 |
| Test Set      | 40            |   3000 |

For data augmentation, we have used the [Augmentor](https://github.com/mdbloice/Augmentor) library in Python.
