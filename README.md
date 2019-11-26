# :bulb: A Guide to Production Level Deep Learning :clapper: :scroll:  :ferry:
[NOTE: This repo is still under development, and any feedback to make it better is welcome :blush: ]

Deploying deep learning models in production can be challenging, as it is far beyond training models with good performance. Several distinct components need to be designed and developed in order to deploy a production level deep learning system (seen below):

<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="95%" height="95%">
</p>

This repo aims to be an engineering guideline for building production-level deep learning systems which will be deployed in real world applications. 

The material presented here is borrowed from [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com) (by [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) at UC Berkeley, [Josh Tobin](http://josh-tobin.com/) at OpenAI, and [Sergey Karayev](https://sergeykarayev.com/) at Turnitin), [TFX workshop](https://conferences.oreilly.com/tensorflow/tf-ca/public/schedule/detail/79327) by [Robert Crowe](https://www.linkedin.com/in/robert-crowe/), and [Pipeline.ai](https://pipeline.ai/)'s [Advanced KubeFlow Meetup](https://www.meetup.com/Advanced-KubeFlow/) by [Chris Fregly](https://www.linkedin.com/in/cfregly/).

The following figure represents a high level overview of different components in a production level deep learning system:
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra_tooling.png" title="" width="95%" height="95%">
</p>
In the following, we will go through each module and recommend toolsets and frameworks as well as best practices from practitioners that fit each component. 

# Full stack pipeline 
## 1. Data Management 
### 1.1 Data Sources 
* Supervised deep learning requires a lot of labeled data
* Labeling own data is costly! 
* Here are some resources for data: 
  * Open source data (good to start with, but not an advantage) 
  * Data augmentation (a MUST for computer vision, an option for NLP)
  * Synthetic data (almost always worth starting with, esp. in NLP)
### 1.2  Data Labeling 
* Requires: separate software stack (labeling platforms), temporary labor, and QC
* Sources of labor for labeling: 
  * Crowdsourcing (Mechanical Turk): cheap and scalable, less reliable, needs QC
  * Hiring own annotators: less QC needed, expensive, slow to scale 
  * Data labeling service companies:
    * [FigureEight](https://www.figure-eight.com/)  
* Labeling platforms: 
  * [Prodigy](https://prodi.gy/): An annotation tool powered
by active learning (by developers of Spacy), text and image 
  * [HIVE](https://thehive.ai/): AI as a Service platform for computer vision  
  * [Supervisely](https://supervise.ly/): entire computer vision platform 
  * [Labelbox](https://labelbox.com/): computer vision  
  * [Scale](https://scale.com/) AI data platform (computer vision & NLP)

    
### 1.3. Data Storage 
* Data storage options: 
  * **Object store**: Store binary data (images, sound files, compressed texts) 
    * [Amazon S3](https://aws.amazon.com/s3/) 
    * [Ceph](https://ceph.io/) Object Store
  * **Database**: Store metadata (file paths, labels, user activity, etc). 
    * [Postgres](https://www.postgresql.org/) is the right choice for most of applications, with the best-in-class SQL and great support for unstructured JSON. 
  * **Data Lake**: to aggregate features which are not obtainable from database (e.g. logs)
    * [Amazon Redshift](https://aws.amazon.com/redshift/)
  * **Feature Store**: store, access, and share machine learning features 
 (Feature extraction could be computationally expensive and nearly impossible to scale, hence re-using features by different models and teams is a key to high performance ML teams). 
    * [FEAST](https://github.com/gojek/feast) (Google cloud, Open Source)
    * [Michelangelo Palette](https://eng.uber.com/michelangelo/) (Uber)
* Suggestion: At training time, copy data into a local or networked **filesystem** (NFS). <sup>[1](#fsdl)</sup> 

### 1.4. Data Versioning 
* It's a "MUST" for deployed ML models:  
  **Deployed ML models are part code, part data**. <sup>[1](#fsdl)</sup>  No data versioning means no model versioning. 
* Data versioning platforms: 
  * [DVC](https://dvc.org/): Open source version control system for ML projects 
  * [Pachyderm](https://www.pachyderm.com/): version control for data 
  * [Dolt](https://www.liquidata.co/): versioning for SQL database 
    
### 1.5. Data Processing 
* Training data for production models may come from different sources, including *Stored data in db and object stores*, *log processing*, and *outputs of other classifiers*.
* There are dependencies between tasks, each needs to be kicked off after its dependencies are finished. For example, training on new log data, requires a preprocessing step before training. 
* Makefiles are not scalable. "Workflow manager"s become pretty essential in this regard.
* **Workflow orchestration:**
  * [Luigi](https://github.com/spotify/luigi) by Spotify
  * [Airflow](https://airflow.apache.org/) by Airbnb: Dynamic, extensible, elegant, and scalable (the most widely used)
      * DAG workflow 
      * Robust conditional execution: retry in case of failure  
      * Pusher supports docker images with tensorflow serving 
      * Whole workflow in a single .py file 

<p align="center">
  <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/airflow_pipe.png" title="" width="65%" height="65%">
   </p>
   

## 2. Development, Training, and Evaluation 
### 2.1. Software engineering
* Winner language: Python
* Editors:
   * Vim
   * Emacs  
   * [VS Code](https://code.visualstudio.com/) (Recommended by the author): Built-in git staging and diff, Lint code, open projects remotely through ssh 
   * Notebooks: Great as starting point of the projects, hard to scale (fun fact: Netflix’s Notebook-Driven Architecture is an exception, which is entirely based on [nteract](https://nteract.io/) suites). 
      * [nteract](https://nteract.io/): a next-gen React-based UI for Jupyter notebooks
      * [Papermill](https://github.com/nteract/papermill): is an [nteract](https://nteract.io/) library built for *parameterizing*, *executing*, and *analyzing* Jupyter Notebooks.
      * [Commuter](https://github.com/nteract/commuter): another [nteract](https://nteract.io/) project which provides a read-only display of notebooks (e.g. from S3 buckets).
   * [Streamlit](https://streamlit.io/): interactive data science tool with applets
 * Compute recommendations <sup>[1](#fsdl)</sup>:
   * For *individuals* or *startups*: 
     * Development: a 4x Turing-architecture PC
     * Training/Evaluation: Use the same 4x GPU PC. When running many experiments, either buy shared servers or use cloud instances.
   * For *large companies:* 
     * Development: Buy a 4x Turing-architecture PC per ML scientist or let them use V100 instances
     * Training/Evaluation: Use cloud instances with proper provisioning and handling of failures
 * Cloud Providers: 
   * GCP: option to connect GPUs to any instance + has TPUs 
   * AWS:  
### 2.2. Resource Management 
  * Allocating free resources to programs 
  * Resource management options: 
    * Old school cluster job scheduler ( e.g. [Slurm](https://slurm.schedmd.com/) workload manager )
    * Docker + Kubernetes
    * Kubeflow 
    * [Polyaxon](https://polyaxon.com/) (paid features)
    
### 2.3. DL Frameworks 
  * Unless having a good reason not to, use Tensorflow/Keras or PyTorch. <sup>[1](#fsdl)</sup> 
  * The following figure shows a comparison between different frameworks on how they stand for *"developement"* and *"production"*.  

  <p align="center">
  <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/frameworks.png" title="" width="95%" height="95%">
   </p>

  
### 2.4. Experiment management

* Development, training, and evaluation strategy:
  * Always start **simple** 
    * Train a small model on a small batch. Only if it works, scale to larger data and models, and hyperparameter tuning!  
  * Experiment management tools: 
  * [Tensorboard](https://www.tensorflow.org/tensorboard)
      * provides the visualization and tooling needed for ML experimentation  
  * [Losswise](https://losswise.com/) (Monitoring for ML)
  * [Comet](https://www.comet.ml/): lets you track code, experiments, and results on ML projects
  * [Weights & Biases](https://www.wandb.com/): Record and visualize every detail of your research with easy collaboration 
  * [MLFlow Tracking](https://www.mlflow.org/docs/latest/tracking.html#tracking): for logging parameters, code versions, metrics, and output files as well as visualization of the results.
    * Automatic experiment tracking with one line of code in python
    * Side by side comparison of experiments 
    * Hyper parameter tuning 
    * Supports Kubernetes based jobs 
    
### 2.5. Hyperparameter Tuning 
  * Approaches: 
    * Grid search 
    * Random search 
    * Bayesian optimization
    * HyperBand

  * Platforms: 
    * [Katib](https://github.com/kubeflow/katib): Kubernete's Native System   for Hyperparameter Tuning and Neural Architecture Search, inspired by   [Google vizier](https://static.googleusercontent.com/media/ research.google.com/ja//pubs/archive/  bcb15507f4b52991a0783013df4222240e942381.pdf) and supports multiple ML/DL   frameworks (e.g. TensorFlow, MXNet, and PyTorch). 
    * [Hyperas](https://maxpumperla.com/hyperas/): a simple wrapper around  hyperopt for Keras, with a simple template notation to define  hyper-parameter ranges to tune.
    * [SIGOPT](https://sigopt.com/):  a scalable, enterprise-grade  optimization platform 
    * [Ray-Tune](https://github.com/ray-project/ray/tree/master/python/ray/ tune): A scalable research platform for distributed model selection (with  a focus on deep learning and deep reinforcement learning) 
    * [Sweeps](https://docs.wandb.com/library/sweeps) from [Weights & Biases] (https://www.wandb.com/): Parameters are not explicitly specified by a   developer. Instead they are approximated and learned by a machine   learning model.
    * [Keras Tuner](https://github.com/keras-team/keras-tuner): A hyperparameter tuner for Keras, specifically for tf.keras with TensorFlow 2.0.

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
   * Types of tests: 
     * Training system tests: testing training pipeline
     * Validation tests: testing prediction system on validation set 
     * Functionality tests: testing prediction system on few important examples 
* Continuous Integration: Running tests after each new code change pushed to the repo 
 * SaaS for continuous integration: 
    * [Argo](https://argoproj.github.io/): Open source Kubernetes native workflow engine for orchestrating parallel jobs (incudes workflows, events, CI and CD).
    * [CircleCI](https://circleci.com/): Language-Inclusive Support, Custom Environments, Flexible Resource Allocation, used by instacart, Lyft, and StackShare.
    * [Travis CI](https://travis-ci.org/)
    * [Buildkite](https://buildkite.com/): Fast and stable builds, Open source agent runs on almost any machine and architecture, Freedom to use your own  tools and services
    * Jenkins: Old school build system  


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
         * SaaS solutions
            * [Seldon](https://www.seldon.io/): serve and scale models built in any framework on Kubernetes
            * [Algorithmia](https://algorithmia.com/)
   * Decision making: CPU or GPU? 
      * CPU inference:
         * CPU inference is preferable if it meets the requirements.
         * Scale by adding more servers, or going serverless. 
      * GPU inference: 
         * TF serving or Clipper 
         * Adaptive batching is useful 
  * (Bonus) Deploying Jupyter Notebooks:
      * [Kubeflow Fairing](https://github.com/kubeflow/fairing) is a hybrid deployment package that let's you deploy your *Jupyter notebook* codes! 
    
### 4.5 Service Mesh and Traffic Routing 
* Transition from monolithic applications towards a distributed microservice architecture could be challenging. 
* A **Service mesh** (consisting of a network of microservices) reduces the complexity of such deployments, and eases the strain on development teams.
  * [Istio](https://istio.io/): a service mesh to ease creation of  a network of deployed services with load balancing, service-to-service authentication, monitoring, with few or no code changes in service code. 
### 4.4. Monitoring:
* Purpose of monitoring: 
   * Alerts for downtime, errors, and distribution shifts 
   * Catching service and data regressions 
* Cloud providers solutions are decent 
* [Kiali](https://kiali.io/):an observability console for Istio with service mesh configuration capabilities. It answers these questions: How are the microservices connected? How are they performing?

#### Are we done?
<p align="center">
   <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/post-deploy.png" title="" width="65%" height="65%">
</p>

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
### 4.6. All-in-one solutions
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
   <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra-cmp.png" title="" width="100%" height="100%">
</p>

# Tensorflow Extended (TFX) 

<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/tfx_config.png" title="" width="95%" height="95%">
</p>

# Airflow and KubeFlow ML Pipelines 

<p align="center">
    <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/kubeflow_pipe.png" title="" width="45%" height="45%">
</p>


## Other useful links: 
* [Lessons learned from building practical deep learning systems](https://www.slideshare.net/xamat/lessons-learned-from-building-practical-deep-learning-systems)
* [Machine Learning: The High Interest Credit Card of Technical Debt](https://ai.google/research/pubs/pub43146)
 
## [Contributing](https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/CONTRIBUTING.md)

## References: 

<a name="fsdl">[1]</a>: [Full Stack Deep Learning Bootcamp](https://fullstackdeeplearning.com/), Nov 2019. 

<a name="pipe">[2]</a>: [Advanced KubeFlow Workshop](https://www.meetup.com/Advanced-KubeFlow/) by [Pipeline.ai](https://pipeline.ai/), 2019. 

<a name="pipe">[3]</a>: [TFX: Real World Machine Learning in Production](https://cdn.oreillystatic.com/en/assets/1/event/298/TFX_%20Production%20ML%20pipelines%20with%20TensorFlow%20Presentation.pdf)

   
    
