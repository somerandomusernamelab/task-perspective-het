# Task-Perspective-Het

## 📌 Overview

This repository contains the official implementation of the paper:

**Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions**\
Kasra Borazjani, Payam Abdisarabshali, Naji Khosravan, Seyyedali Hosseinalipour\
ArXiv Preprint, 2025

Federated Learning (FL) enables decentralized model training while preserving data privacy, but its performance is significantly impacted by the **non-IID nature** of data across clients. Traditionally, non-IID settings in FL have been defined based on **label distribution skew**. However, in this work, we demonstrate that this definition oversimplifies real-world heterogeneity, particularly in computer vision tasks beyond classification.

We introduce a **task-specific perspective** on data heterogeneity, leveraging **embedding-based distributions** rather than label distributions to model non-IID data. By extracting **task-specific data embeddings** using pre-trained deep neural networks, we redefine FL data heterogeneity at a more fundamental level. Our methodology clusters data points in embedding space and distributes them among clients using a **Dirichlet allocation strategy**, better reflecting the diversity encountered in real-world FL applications.

Through extensive experiments, we show that current FL approaches **overestimate performance** when relying solely on label distribution skew. We provide new benchmark performance measures and highlight open research directions to further advance FL in non-IID settings.

### 🔥 Key Contributions

- We challenge the conventional **label-based definition** of non-IID data in FL and propose a **task-specific embedding-based perspective**.
- We introduce a **new data partitioning methodology** that clusters data in embedding space, resulting in a more realistic non-IID distribution.
- We evaluate multiple FL algorithms under this setting and show that existing methods **struggle more than previously assumed**, exposing an important gap in the literature.
- We establish **new benchmark performance measures** for FL models under embedding-based heterogeneity and outline key research challenges.

---

## 📂 Repository Structure

```
task-perspective-het/
│── datasets/            # Dataset and preprocessing scripts
│── models/              # Model implementations
│── network/             # Training and evaluation scripts
│── utils/               # Helper functions and utilities
│── results/             # Logs, figures, and saved models
│── README.md            # This file
│── requirements.txt     # Required dependencies
│── extract_datapoints_for_experiments.py    # Extracts the datapoints that will be used in the experiments
|── extract_embeddings.py # Used to extract embeddings from trained centralized models
|── losses.py # Defining the losses used in training the tasks
|── model_initialization # Used to create the initial model weights used for each task
|── test_fed.py  # Testing the federated performance of a specific task
|── train_centralized_task_model.py # Training each task's centralized model to extract embeddings from
```

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/somerandomusernamelab/task-perspective-het.git
cd task-perspective-het
pip install -r requirements.txt
```

You will also have to install the CLIP module from the source using (prerequisites are included in the requirements.txt file):

```bash
pip install git+https://github.com/openai/CLIP.git
```



### Dataset

We have used the Taskonomy dataset in our experiments. To see how to download the dataset, please refer to the manual available [here](https://github.com/StanfordVL/taskonomy/tree/master/data). You can also use the alternative sample we have provided below to recreate our results:

```bash
# Make sure everything is installed
sudo apt-get install aria2
pip install 'omnidata-tools' # Just to make sure it's installed

# Install the 'debug' subset of the Replica and Taskonomy components of the dataset
omnitools.download rgb normal class_scene keypoints3d depth_euclidean reshading segment_semantic \
  --components taskonomy \
  --subset tiny \
  --name <insert name> --email <insert email> \
  --dest ./taskonomy_dataset/ --agree-all
```

## 🏋️ Training

To train the centralized models, run:

```bash
python main.py --config configs/config.yaml
```

Depending on the dataset and configuration, training time may vary. We recommend using a **GPU-enabled environment** for optimal performance.

## 📊 Evaluation

Run the evaluation script to reproduce results:

```bash
python evaluate.py --model_path path/to/checkpoint
```

Ensure that the model checkpoint is correctly specified in the argument. Evaluation results, including **accuracy, loss trends, and task-specific performance metrics**, will be logged and saved.

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{your_paper,
  author = {Kasra Borazjani, Payam Abdisarabshali, Naji Khosravan, Seyyedali Hosseinalipour},
  title = {Redefining non-IID Data in Federated Learning for Computer Vision Tasks: Migrating from Labels to Embeddings for Task-Specific Data Distributions},
  journal = {ArXiv Preprint},
  year = {2025},
  doi = {https://doi.org/10.48550/arXiv.2503.14553}
}
```

## 📌 Contact

For questions, feel free to open an issue or contact: [Your Email / Website / Social Media]

---

Let me know if any other details need to be adjusted!

