# DeepCovid Dataset and Implementation in PyTorch

In this repository, we provide the PyTorch implementation of the DeepCovid Framework (the training and inference code) for the research community to use.

We also provide a labeled dataset of 5000 images, which is used to train and evaluate this model.



## COVID-XRay-5K DATASET
We prepared a dataset of around 5000 images, which can be downloaded from here: [dataset_link](https://www.dropbox.com/s/mzas2tkd80pubh7/data_covid5k.zip?dl=0)

Two sources are used to create this dataset:
* [Covid-Chestxray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset), for COVID-19 X-ray samples
* [ChexPert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/), for Non-COVID samples

COVID-19 samples from Covid-Chestxray-Dataset are extracted from a several publications, and it is important to verify all their labels. With the help of a board-certified radiologist, we went through X-ray images of COVID-19 samples, and only kept those which were selceted to have a clear sign of COVID-19 by our radiologist. We also only kept the posterior-anterior images. 

For Non-COVID samples, we tried to uniformly sample images from ChexPert. More details on the dataset are provided in our [paper](https://arxiv.org/pdf/2004.09363.pdf).

Some of the sample images from our dataset are shown below. The images in the first row show COVID-19 cases, and the images in the remaining rows denote non-COVID cases.

![samples](https://github.com/shervinmin/DeepCovid/blob/master/results/covid5k_samples.png)

As the number of COVID-19 samples are much fewer than the number of Non-COVID samples, we used several data-augmentation techniques (as well as over-sampling) to increase the number of COVID-19 samples in training, to have a less imbalanced training set. Hopefully more cleanly labeled X-ray images from COVID-19 cases become available soon, so we do not have this imbalanced data issue.

For data augmentation, we have used the [Augmentor](https://github.com/mdbloice/Augmentor) library in Python.

## Training a model
We have provided a Python script to train a ResNet18 model on the training data. 
The training script gets a few arguments from the user, such as the training data path, leanring rate, number of epochs, etc. There is a default value for all of these arguments, but if you can specify your own argument too. 

**You can run the training code as:**

```
python ResNet18_train.py --dataset_path ./data/ --batch_size 20 --epoch 50 --num_workers 4 --learning_rate 0.001
```

This code fine-tunes a pre-trained ResNet18 model on the training dataset. 

The architecture of ResNet18 is shown below:

![resnet18](https://github.com/shervinmin/DeepCovid/blob/master/results/resnet18.png)


Note that if you are running this on Windows, you need to set the num_workers to 0, as PyTorch support on Windows is still limited.

## Inference Code
We have also provided the code for doing inference on the trained models. Given the path for the test samples, the inference code provides the predicted scores (probabilities) and predicted labels of the samples. 
It also provides the sensitivity and specificity rate for different cut-off threshold.

In addition, the hisotgram of the predicted probabilities, the convusion matrix, and ROC curve are also generated in the inference code. 

**The inference code can be as:**

```
python inference.py --test_covid_path ./data/test/covid/ --test_non_covid_path ./data/test/non/ --trained_model_path ./models/covid_resnet18_epoch100.pt
```


The predicted scores of the finetuned ResNet18 model on the test set are as below.
As we can see the predicted probability scores of the COVID-19 samples are much larger than those of non-Covid samples, which is encouraging.

![ResNet_scores](https://github.com/shervinmin/DeepCovid/blob/master/results/hist_resnet18.png)


## Disclaimer
Although the results derived by finetuning various convolutional networks for COVID-19 detection are really encouraging, it is worth to mention that due to the limited number of COVID-19 images in our training and test sets, study on a larger number of cleanly labeled COVID-10 images is required to have a more concrete conclusion on the potential of X-ray images for COVID-19 deteciton.

## Usage Right:
This work is done by Shervin Minaee, Milan Sonka (the previous editor in chief of IEEE TMI), Rahele Kafieh, Shakib Yazdani, and Ghazaleh Jamalipour Soufi (our radiologist). The Arxiv version of the paper can be downloaded from [here](https://arxiv.org/pdf/2004.09363.pdf):
If you find this work useful, you can refer our work as:

```
@article{minaee2020deep,
  title={Deep-COVID: Predicting COVID-19 From Chest X-Ray Images Using Deep Transfer Learning},
  author={Minaee, Shervin and Kafieh, Rahele and Sonka, Milan and Yazdani, Shakib and Jamalipour Soufi, Ghazaleh},
  journal={arXiv preprint arXiv:2004.09363},
  year={2020}
}
```
