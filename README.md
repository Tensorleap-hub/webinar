
# YOLOv7 model with Cognata and Foresight datasets
  
The Foresight dataset is a collection of images where taken with ADAS and front parking cameras that are installed in 
the front of vehicles.
The [YOLOv7](https://github.com/WongKinYiu/yolov7) model is an object detection trained on `Cognata` <need to change to foresight> dataset.
A new unlabeled client data in production `foresight_unlabeled` <*need to change?> detected to shift from our trained data distribution using Tensorleap (TL). Here we demonstrate how to detect such issues and handling by tuning a data synthesizing process.

First, by TL's unsupervised algorithms we automatically identify data gaps of unseen datasets and without the need of labels.
Then, we generate data based on statistics and insights we analyze in TL. We test again to identify if our generated data distribute accordingly to our target set.
Iteratively, we improve the generated data until satisfying convergence. When we have the final generated dataset, we can fine tune our model on the data distributes as our target data which is unlabeled and consists too few samples.

### Data Shift Detection

We identify the data shift using two strategies in the platform:

1. In TL's insights panel, the new samples detected as `under-represented` cluster. 
![](images/insight_shift_detected.png)

2. From the Population Exploration plot, it's evident that the new samples are geometrically distinct from the original data.
![](images/origin_target_samples.png)

The unlabeled data is sourced from a different camera captured by a fisheye camera. The model exhibits errors, misclassifies objects, particularly pedestrians:
![](images/sample_error_1.png)
![](images/sample_error_2.png)
![](images/sample_error_3.png)

### Samples Generation I 

We synthesize samples (CognataA) to tune our model targeting our unlabeled sample. However, there is still a major data shift as seen in the latent space:

![](images/sample_generation_1.png)

We are also alarmed by TL insights: the generated data identified as underrepresented cluster additionally to the target sample:

![](images/sample_generation_1_insights.png)

### Correlated Metadata 

Analyzing external metadata variables, we observe a correlation to `red channel std`: the generated sample has higher values compared to the target data:

![](images/PE_red_std.png) Samples colored by red channel std level 
![](images/dash_red_std.png) Red channel std across the sources 


### Samples Generation II

* Accordingly, we generate new images (`CognataB`) with a lower std of the red channel.

![](images/PE_sample_generation_2.png) Latent space contains the new generated data (`CognataB`)
![](images/PE_red_std_sample_generation_2.png) Samples colored by red channel std level
![](images/dash_red_std_sample_generation_2.png) Red channel std across the sources including new generated data


### Data Quality Evaluation

New synthesized samples (in green) are more closely aligned to the target data (in yellow):

![](images/PE_gen_quality_eval.png) 

Now we can tune it by selecting using a threshold of distance from the target centroid for instance, or by using an image feature metadata that is correlated to the distance. For instance, using 'color temperature' as seen in below. We can iteratively generate the samples while tuning the image 'color temperature' until reaching satisfactory convergence.

![](images/PE_gen_quality_prioritization_color_temp.png) Samples colored by `color tempratue` level



**Tensorleap** helps to explore the latent space of a dataset and find new and important insights. It can also be used 
to find unlabeled clusters or miss-labeling in the dataset in the most efficient way.
This quick start guide will walk you through the steps to get started with this example repository project.

Since, our target data is unlabeled and consist of few samples we synthesize data with the statistics we found which characterize our target data compared to the origin.




* Through PE, we observe that although the samples spread in the latent space closer to the target, yet, the distribution more closely aligns with the target's outliers. Moreover, further insights revealed of underrepresented samples. Consequently, we enhance the generation process.
* Investigating other metadata variables that are correlated to the target samples we identify `red_channel_mean`. The target samples have lower values compares to our generated samples.



### Population Exploration

Below is a population exploration plot. It contains samples that are represented based on the model's latent space, built using the extracted features. 
Using Tensorleap we can cluster the latent space by kmeans algorithm. Below we can see that cluster number 4 contains images of driving at night hours, cluster number 6 contained urban images.

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

![with_cognata_new](images/with_cognata_new.png)

When we look at the population exploration plot above we can see that the synthetic data does not fit. 
Looking the dashboard we can see that the average image std has a big gap between the datasets.

![image_std](images/image_std.png)

After changing and simulate another data we can see that it fits to the target distribution. We also see that we 
do not need all the simulated data, only the images that their distribution closed to the target data distribution.

![PE_with_cognata](images/PE_with_cognata.png)

Looking at the image std VS origin leads to the same conclusions.

![image_std_with_coganta](images/image_std_with_coganta.png)

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




