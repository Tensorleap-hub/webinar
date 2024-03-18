
# YOLOv7 model with Cognata and Foresight datasets

The Foresight dataset is a collection of images where taken with ADAS and front parking cameras that are installed in 
the front of vehicles.
The [YOLOv7](https://github.com/WongKinYiu/yolov7) model is a neural
network model that can be trained to predict object detection given images.

**Tensorleap** helps to explore the latent space of a dataset and find new and important insights. It can also be used 
to find unlabeled clusters or miss-labeling in the dataset in the most efficient way.
This quick start guide will walk you through the steps to get started with this example repository project.

### Population Exploration

Below is a population exploration plot. It represents a samples similarity map based on the model's latent space,
built using the extracted features of the trained model.

Using Tensorleap we can cluster the latent space by kmeans algorithm. Below we can see that cluster number 4 contains 
images of driving at night hours, cluster number 6 contained urban images.

#### cluster number 4:

<div style="display: flex">
  <img src="images/cluster_4_1.png" alt="Image 1" style="margin-right: 10px;">
  <img src="images/cluster_4_2.png" alt="Image 2" style="margin-left: 10px;">
</div>

#### cluster number 6:

<div style="display: flex">
  <img src="images/cluster_6_1.png" alt="Image 1" style="margin-right: 10px;">
  <img src="images/cluster_6_2.png" alt="Image 2" style="margin-left: 10px;">
</div>


When we add the target set and colored the dots (samples) by 'origin set', it shows a visualization of the original 
foresight dataset (pink) and a new client dataset (purple). 
We can see two distinct clusters, this means there is a difference in their representation, the training set distribution    
does not fit to the test distribution. 
The difference can be due to the differences in camera degree, camera location, the position of the steering wheel 
in the car, the weather and so on.

![Latent space_dataset_state](images/Latent_space_dataset_state.png)

When we change the dots size to be determined by the loss predicted-it means the loss value tensorleap predict that 
the model will predict we can see that the target data has much bigger predicted loss. It means the possibility the 
model will fail to correctly predict the gt bounding boxes is high.

![Latent space_dataset_state_predicted_loss](images/Latent_space_dataset_state_predicted_loss.png)

To address this problem there is a need to generate new synthetic data and train the model on it. 
With the help of 'Cognata' a vast amount of labeled data was simulated using realistic sensor setup.
Using Tensorleap we can validate if the simulation data comes from the target distribution in the eye of the model.

![with_cognata_new](images/with_cognata_new_old.png)

When we look at the population exploration plot above we can see that the synthetic data does not fit. 
Looking the dashboard we can see that the average image std has a big gap between the datasets.

![image_std](images/image_std_old.png)

After changing and simulate another data we can see that it fits to the target distribution. We also see that we 
do not need all the simulated data, only the images that their distribution closed to the target data distribution.

![PE_with_cognata](images/PE_with_cognata_old.png)

Looking at the image std VS origin leads to the same conclusions.

![image_std_with_coganta](images/image_std_with_coganta_old.png)

# Getting Started with Tensorleap Project

This quick start guide will walk you through the steps to get started with this example repository project.

## Prerequisites

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher).
- **[Poetry](https://python-poetry.org/)**.
- **[Tensorleap](https://tensorleap.ai/)** platform access. To request a free trial click [here](https://meetings.hubspot.com/esmus/free-trial).
- **[Tensorleap CLI](https://github.com/tensorleap/leap-cli)**.


## Tensorleap **CLI Installation**

with `curl`:

```
curl -s https://raw.githubusercontent.com/tensorleap/leap-cli/master/install.sh | bash
```

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorleap:

```
tensorleap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: https://api.CLIENT_NAME.tensorleap.ai/api/v2

<br>

**How To Generate CLI Token from the UI**

1. Login to the platform in 'CLIENT_NAME.tensorleap.ai'
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN` in the bottom-left corner.
3. Once a CLI token is generated, just copy the whole text and paste it into your shell.


## Tensorleap **Project Deployment**

To deploy your local changes:

```
leap project push
```

### **Tensorleap files**

Tensorleap files in the repository include `leap_binder.py` and `leap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**leap.yaml**

leap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment.

For any additional file being used, we add its path under the `include` parameter:

```
include:
    - leap_binder.py
    - squad_albert/configs.py
    - [...]
```

**leap_binder.py file**

`leap_binder.py` configures all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run `leap_test.py` file using poetry:

```
poetry run test
```

This file will execute several tests on leap_binder.py script to assert that the implemented binding functions: preprocess, encoders,  metadata, etc,  run smoothly.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*




