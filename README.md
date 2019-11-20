# Production-Level-Deep-Learning
Deploying deep learning models in production could be challenging, as it's far beyond just training models with good perfromance [Under developement].    

<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="75%" height="75%">
</p>


This repo aims to serve as a an engieering guideline for building production-level deep learning systems to be deployed in real world applications. * Most of the content is borrowed from [Full Stack Deep Learning](https://fullstackdeeplearning.com) and Pipeline.ai [https://pipeline.ai/]'s [Advanced KubeFlow Meetup](https://www.meetup.com/Advanced-KubeFlow/) * 

The following figure represent different components in a production level deep learning system:
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra_tooling.png" title="" width="95%" height="95%">
</p>

In the following, we will go through each module and recommend toolsets and frameworks as well as best practices from practioners that fit each component. 

## 1. Data Management 
### 1. Data Sources 
    * Open source 
    * Data augmentation 
    * Synthetic data 
### 2. Labeling 
    * Platforms: 
        * Prodigy 
        * HIVE 
        * Supervisely 
        * Labelbox 
        * Scale.ai 
    * Sources of labor: 
        * Crowdsource 
        * Service companies 
            * FigureEight 
        * Hire annotators 
    
### Storage 
    * **object store**: Store binary data (images, sound files, compressed texts) 
        * Aamzon S3 
        * Ceph 
    * **Database**: Store metadata (file paths, labels, user activity, etc). 
        * Postgres is the right choice for most of applications, with the best-in-class SQL and great support for unstructured JSON. 
    * **Data Lake**: to aggregate features which are not obtainable from database (e.g. logs)
    * **Feature Store**: [TBC]
    * Train time: copy data into a local or netwroked **filesystem** 
### Versioning 
    * DVC: Open source version control system for ML projects 
    * Pachyderm: version control for data 
    * Dolt: versioning for SQL database 
    
### Processing 
- Training data may come from different sources: Stored data in db and object stores, log processing, outputs of other classifiers.
- There are dependencies between tasks, one needs to kick off after its dependencies are finished. 
* Workflows: 
  ** Airflow (most commonly used)

## 2. Development, Training, and Evaluation 
### Software engineering
 * Editors:
   * Vim 
   * VS Code 
     * Built in git staging and diff, Lint code, open projects remotely through ssh 
   * Jupyter Notebooks: Great as starting points of the projects, hard to scale 
   * Streamlit: interactive applets 
 * Compute recommendations:
   * for solo/startup: 
     * Development: a 4x Turing-architecture PC
     * Training/Evaluation: Use the same 4x GPU PC. When running many experiments, either buy shared servers or use cloud instances.
   * for larger companies: 
     * Development: Buy a 4x Turing-architecture PC per ML scientist or let them use V100 instances
     * Training/Evaluation: Use cloud instances with proper provisioning and handling of failures
### Resource Management 
  * allocating free resources to programs 
  * Old school cluster job scheduler (Slurm)
  * Docker + Kubernetes
  * Kubeflow 
  * Polyaxon (paid features)
  
### DL Frameworks 
  * Unless having a good reason not to, use Tensorflow/Keras or PyTorch
  <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/frameworks.png" title="" width="50%" height="50%">
  
### Experiment management
  * Tensorboard 
  * Losswise (Monitoring for ML)
  * Comet.ml 
  * Weights & Biases
  * MLFlow tracking 
  
### Hyperparameter Tuning 
  * Hyperas 
  * SIGOPT 
  * Ray - Tune 
  * Weights & Biases
### Distributed Training 
  * Data parallelism: Use it when iteration time is too long (both tensorflow and PyTorch support)
  * Model parallelism: when model does not fit on a single GPU 
  * Other solutions: 
    * Ray 
    * Horovod

## 3. Troubleshooting [TBD]

## 3. Testing and Deployment 
### Testing and CI/CD
Machine Learning production software requires a more diverse set of test suites than traditional software:
<p align="center">
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/testing.png" title="" width="95%" height="95%">
</p> 
  * Unit and Integration Testing: 
      * Types of test: 
        * Training system tests: testing training pipeline
        * Validation tests: testing prediction system on validation set 
        * Functionality tests: testing prediction system on few important examples 
  * Continous Integration: Running tests after each new code change pushed to the repo 
    * SaaS for continous intergation: 
      * CircleCI, Travis 
      * Jenkins, Buildkite


### Web Depolyment
  * Consists of a **Prediction System** and a **Serving System**
      * Prediction System: Process input data, make predictions 
      * Serving System (Web server): 
        * Serve prediction with scale in mind  
        * Use REST API to serve prediction HTTP requests
        * Calls the prediction system to respond 
  * Serving system options: 
      * 1. Deploy to VMs, scale by adding instances 
      * 2. Deploy as containers, scale via orchestration 
          * Containers 
              * Docker 
          * Container Orchestration:
              * kubernetes (the most popular now)
              * MESOS 
              * Marathon 
      * 3. Deploy as a "serverless function"
      * 4. Deploy via a model serving solution 
  
### Model Conversion 
  * Open Neural Network Exchange (ONNX): open-source format for deep learning models 

### Monitoring 
### Hardware
  * On Device  
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
## All-in-one solutions
   * Michelangelo (Uber)
   * Tensorflow Extended (TFX)
   * Google Cloud AI Platform 
   * Amazon SageMaker 
   * Neptune 
   * FLOYD 
   * Paperspace 
   * Determined AI 
   * Domino data lab 
<p align="center">
   <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra-cmp.png" title="" width="50%" height="50%">
</p>
   
    
