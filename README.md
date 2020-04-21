# DeepCovid Dataset and Implementation in PyTorch

Here we provide the PyTorch implementation of the DeepCovid Framework (the training and inference code) for the research community to use.



# COVID-XRay-5K DATASET
We prepared a dataset of around 5000 images, for 


# Training a model
We have provided a Python script to train a ResNet18 model on the training data. 
The training script gets a few arguments from the user, such as the training data path, leanring rate, number of epochs, etc. There is a default value for all of these arguments, but if you can specify your own argument too. 
You can run the training code as:

python ResNet18_train.py --dataset_path ./data/ --batch_size 20 --epoch 2 --num_workers 0 --learning_rate 0.0005 --momentum 0.5
