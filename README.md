# xlingual-robustness
Paper: Cross-Lingual Transfer Robustness to Lower-Resource Languages on Adversarial Datasets

# Finetuning mBERT and XLM-R on NER and section title prediction tasks
This work covers generating **main** and **perturbation** datasets and applying them in both **native** and **cross-lingual transfer** settings for **NER** and **section title prediction** tasks on **mBBERT** and **XLMR** models.


## Prerequisites
Create a new virtual environment and install the required packages. Commands to do so below. 

conda create -n transformers python pandas tqdm \
conda activate transformers\
pip install simpletransformers\
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia\
conda install -c anaconda scipy\
conda install -c anaconda scikit-learn\
pip install transformers\
pip install tensorboardx\


# Pipeline

## NER 

Run <mark>train_test_before_bias.ipynb</mark> to find the model performance on original datasets. On this file, the overlap between datasets are computed.

Run <mark>Genarate_perturbation_dataset.ipynb</mark> to generate the prturbation dataset.

Run <mark>train_test_after_bias.ipynb</mark> to find the model performance on perturbation datasets.

## WikiTitle
Run <mark>Genarate_WikiTitle_dataset.ipynb</mark> to generate WikiTitle dataset. Also, you can directly find the dataset from [Datasets/WikiTitle](https://drive.google.com/drive/folders/1sTHfJiYsk9Wq7g3uaDmTgHVy703BdSL7?usp=drive_link).

Run <mark>train_test_before_bias.ipynb</mark> to find the model performance on original datasets.

Run <mark>Genarate_perturbation_dataset.ipynb</mark> to generate the prturbation dataset.

Run <mark>train_test_after_bias.ipynb</mark> to find the model performance on perturbation datasets.

Run <mark>WikiTitle.ipynb</mark> to find the overlap between languages datasets by different considerations like overlap of titles, all section texts, or 128-tokens texts. We used 128-tokens texts for the results.

## Plot
Run <mark>plots.ipynb</mark> to draw plots for both NER and WikiTitle.

