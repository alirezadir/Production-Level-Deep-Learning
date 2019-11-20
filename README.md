# Production-Level-Deep-Learning
Deploying deep learning models in production could be pretty challenging, as it's far beyond training models which acheieve good perfromance.    


<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/components.png" title="" width="75%" height="75%">

This repo aims to serve as a an engieering guideline for building production-level deep learning systems to be deployed in real world applications. [Most of the material is borrowed from [Full Stack Deep Learning](https://fullstackdeeplearning.com)]

The following figure represent different components in a production level deep learning system:
<img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra_tooling.png" title="" width="95%" height="95%">

We will go through each module and introduce toolsets that fit each the best. 

## 1. Data 
### Storage 
### Processing 
### Database 
### Versioning 
### Labeling 

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
  


## 3. Deployment 
### Web Serving 
### Model Conversion 
  * Open Neural Network Exchange (ONNX): open-source format for deep learning models 
### CI/CD and Testing 
  * Unit Testing 
  * Types of test: 
    * Training system tests 
    * validation tests 
    * Functionality tests 
  * Continous Integration 
    * SaaS for continous intergation: 
      * CircleCI, Travis 
      * Jenkins, Buildkite
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
   <img src="https://github.com/alirezadir/Production-Level-Deep-Learning/blob/master/images/infra-cmp.png" title="" width="50%" height="50%">
   
    
