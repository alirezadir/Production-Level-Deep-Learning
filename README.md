# A Guideline to Production Level Deep Learning [*Under developement*]
Deploying deep learning models in production could be challenging, as it's far beyond just training models with good perfromance. As you can see in the following figure, there are several components that need to be properly designed and developed in order to deploy a production level deep learning system:

<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="85%" height="85%">
</p>

This repo aims to serve as a an engineering guideline for building production-level deep learning systems to be deployed in real world applications. 

(*The material presented here is moslty borrowed from [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com) (by [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Josh Tobin](http://josh-tobin.com/), and [Sergey Karayev](https://sergeykarayev.com/)), [TFX workshop](https://conferences.oreilly.com/tensorflow/tf-ca/public/schedule/detail/79327) by [Robert Crowe](https://www.linkedin.com/in/robert-crowe/), and [Pipeline.ai](https://pipeline.ai/)'s [Advanced KubeFlow Meetup](https://www.meetup.com/Advanced-KubeFlow/) by [Chris Fregly](https://www.linkedin.com/in/cfregly/).* )

The following figure represent a high level overview of different components in a production level deep learning system:
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra_tooling.png" title="" width="95%" height="95%">
</p>
In the following, we will go through each module and recommend toolsets and frameworks as well as best practices from practioners that fit each component. 

## 1. Data Management 
### 1.1. Data Sources 
* Open source data (good to start with, not an advantage) 
* Data augmentation 
* Synthetic data 
### 1.2. Labeling 
* Sources of labor for labeling: 
  * Crowdsourcing 
  * Service companies 
      * [FigureEight](https://www.figure-eight.com/) 
  * Hiring annotators 
* Labeling platforms: 
  * [Prodigy](https://prodi.gy/): An annotation tool powered
by active learning (by developers of Spacy), text and image 
  * [HIVE](https://thehive.ai/): AI as a Service platform for computer vision  
  * [Supervisely](https://supervise.ly/): entire computer vision platform 
  * [Labelbox](https://labelbox.com/): computer vision  
  * [Scale](https://scale.com/) AI data platform (computer vision & NLP)

    
### 1.3. Storage 
* Data storage options: 
  * **Object store**: Store binary data (images, sound files, compressed texts) 
    * [Aamzon S3](https://aws.amazon.com/s3/) 
    * [Ceph](https://ceph.io/) Object Store
  * **Database**: Store metadata (file paths, labels, user activity, etc). 
    * [Postgres](https://www.postgresql.org/) is the right choice for most of applications, with the best-in-class SQL and great support for unstructured JSON. 
  * **Data Lake**: to aggregate features which are not obtainable from database (e.g. logs)
    * [Amazon Redshift](https://aws.amazon.com/redshift/)
  * **Feature Store**: storage and access of machine learning features
    * [FEAST](https://github.com/gojek/feast) (Google cloud, Open Source)
    * [Michelangelo](https://eng.uber.com/michelangelo/) (Uber)
* At train time: copy data into a local or networked **filesystem** 

### 1.4. Versioning 
* [DVC](https://dvc.org/): Open source version control system for ML projects 
* [Pachyderm](https://www.pachyderm.com/): version control for data 
* [Dolt](https://www.liquidata.co/): versioning for SQL database 
    
### 1.5. Processing 
- Training data may come from different sources: Stored data in db and object stores, log processing, outputs of other classifiers.
- There are dependencies between tasks, one needs to kick off after its dependencies are finished. 
* Workflows: 
   * [Airflow]("https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/) (most commonly used)

## 2. Development, Training, and Evaluation 
### 2.1. Software engineering
* Editors:
   * Vim 
   * [VS Code](https://code.visualstudio.com/) (Recommended by the author)
     * Built in git staging and diff, Lint code, open projects remotely through ssh 
   * Jupyter Notebooks: Great as starting point of the projects, hard to scale 
   * [Streamlit](https://streamlit.io/): interactive data science tool with applets
 * Compute recommendations <sup>[1](#fsdl)</sup>:
   * For solo/startup: 
     * Development: a 4x Turing-architecture PC
     * Training/Evaluation: Use the same 4x GPU PC. When running many experiments, either buy shared servers or use cloud instances.
   * For larger companies: 
     * Development: Buy a 4x Turing-architecture PC per ML scientist or let them use V100 instances
     * Training/Evaluation: Use cloud instances with proper provisioning and handling of failures
### 2.2. Resource Management 
  * Allocating free resources to programs 
  * Resource management options: 
    * Old school cluster job scheduler ( e.g. [Slurm](https://slurm.schedmd.com/) workload manager )
    * Docker + Kubernetes
    * Kubeflow 
    * [Polyaxon](https://polyaxon.com/) (paid features)
    
### 2.3. DL Frameworks 
  * Unless having a good reason not to, use Tensorflow/Keras or PyTorch <sup>[1](#fsdl)</sup>

  <p align="center">
  <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/frameworks.png" title="" width="50%" height="50%">
   </p>

  
### 2.4. Experiment management
  * Tensorboard 
  * Losswise (Monitoring for ML)
  * Comet.ml 
  * Weights & Biases
  * MLFlow tracking 
  
### 2.5. Hyperparameter Tuning 
  * Hyperas 
  * SIGOPT 
  * Ray - Tune 
  * Weights & Biases
### 2.6. Distributed Training 
  * Data parallelism: Use it when iteration time is too long (both tensorflow and PyTorch support)
  * Model parallelism: when model does not fit on a single GPU 
  * Other solutions: 
    * Ray 
    * Horovod

## 3. Troubleshooting [TBD]

## 4. Testing and Deployment 
### 4.1. Testing and CI/CD
Machine Learning production software requires a more diverse set of test suites than traditional software:
<p align="center">
  <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/testing.png" title="" width="75%" height="75%">
   </p>
   
* Unit and Integration Testing: 
   * Types of test: 
     * Training system tests: testing training pipeline
     * Validation tests: testing prediction system on validation set 
     * Functionality tests: testing prediction system on few important examples 
* Continuous Integration: Running tests after each new code change pushed to the repo 
 * SaaS for continuous integration: 
   * CircleCI, Travis 
   * Jenkins, Buildkite

### 4.2. Web Depolyment
  * Consists of a **Prediction System** and a **Serving System**
      * Prediction System: Process input data, make predictions 
      * Serving System (Web server): 
        * Serve prediction with scale in mind  
        * Use REST API to serve prediction HTTP requests
        * Calls the prediction system to respond 
  * Serving options: 
      * 1. Deploy to VMs, scale by adding instances 
      * 2. Deploy as containers, scale via orchestration 
          * Containers 
              * Docker 
          * Container Orchestration:
              * Kubernetes (the most popular now)
              * MESOS 
              * Marathon 
      * 3. Deploy code as a "serverless function"
      * 4. Deploy via a **model serving** solution
  * Model serving:
      * Specialized web deployment for ML models
      * Batches request for GPU inference 
      * Frameworks:
         * Tensorflow serving 
         * MXNet Model server 
         * Clipper (Berkeley)
         * SaaS solutions (Seldon, Algorithma)
   * Decision making: 
      * CPU inference:
         * CPU inference is preferable if it meets the requirements.
         * Scale by adding more servers, or going serverless. 
      * GPU inference: 
         * TF serving or Clipper 
         * Adaptive batching is useful 
### 4.4. Monitoring:
* Purpose: 
   * Alerts for downtime, errors, and distribution shifts 
   * Catching service and data regressions 
* Cloud providers solutions are decent 


### 4.5. Deploying on Embedded and Mobile Devices  
* Main challenge: memory footprint and compute constraints 
* Solutions: 
   * Quantization 
   * Reduced model size 
      * MobileNets 
   * Knowledge Distillation 
      * DistillBERT (for NLP)
* Embedded and Mobile Frameworks: 
   * Tensorflow Lite
   * PyTorch Mobile
   * Core ML 
   * ML Kit 
   * FRITZ 
   * OpenVINO
* Model Conversion:
   * Open Neural Network Exchange (ONNX): open-source format for deep learning models 
## 4.6. All-in-one solutions
   * Tensorflow Extended (TFX)
   * Michelangelo (Uber)
   * Google Cloud AI Platform 
   * Amazon SageMaker 
   * Neptune 
   * FLOYD 
   * Paperspace 
   * Determined AI 
   * Domino data lab 
<p align="center">
   <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra-cmp.png" title="" width="85%" height="85%">
</p>

## Other useful links: 
* [Lessons learned from building practical deep learning systems](https://www.slideshare.net/xamat/lessons-learned-from-building-practical-deep-learning-systems)
 
## [Contributing](https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/CONTRIBUTING.md)

## References: 

<a name="fsdl">[1]</a>: [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/)

<a name="pipe">[2]</a>: [Advanced KubeFlow Workshop](https://www.meetup.com/Advanced-KubeFlow/) by [Pipeline.ai](https://pipeline.ai/)

<a name="pipe">[3]</a>: [TFX: Real World Machine Learning in Production](https://cdn.oreillystatic.com/en/assets/1/event/298/TFX_%20Production%20ML%20pipelines%20with%20TensorFlow%20Presentation.pdf)

   
    
