# Vertical Attention Network: an end-to-end model for handwritten text recognition at paragraph level.
This repository is a public implementation of the paper: ""

It focuses on Optical Character Recognition (OCR) applied at line and paragraph levels.

We obtained the following results at line level:

|  Dataset  |  cer |  wer  |
|:------------:|:----:|:-----:|
|      IAM     | 4.95 | 18.73 |
|     RIMES    | 3.19 | 10.25 |
|   READ2016   | 4.28 | 19.71 |
| ScribbleLens | 6.66 | 25.12 |

For the paragraph level, here are the results:

| Dataset  |  cer |  wer  |
|:------------:|:----:|:-----:|
|      IAM     | 4.32 | 16.24 |
|     RIMES    | 1.90 | 8.83 |
|   READ2016   | 3.63 | 16.75 |


Table of contents:
1. [Getting Started](#Getting Started)
2. [Datasets](#Datasets)
3. [Training And Evaluation](#Training)

## Getting Started
Implementation has been tested with Python 3.6.

Clone the repository:

```
git clone https://github.com/FactoDeepLearning/VerticalAttentionOCR.git
```

Install the dependencies:

```
pip install -r requirements.txt
```


## Datasets
This section is dedicated to the datasets used in the paper: download and formatting instructions are provided 
for experiment replication purposes.

### IAM

#### Details

IAM corresponds to english grayscale handwriting images (from the LOB corpus).
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 6,482 |     976    | 2,915 |
| paragraph |  747  |     116    |  336  |

#### Download

<ul>
<li>Register at the [FKI's webpage](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php).
<li>Download the dataset [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)</li>
<li>Move the following files into the folder Datasets/raw/IAM/
    <ul>
        <li>formsA-D.tgz</li>
        <li>formsE-H.tgz</li>
        <li>formsI-Z.tgz</li>
        <li>lines.tgz</li>
        <li>ascii.tgz</li>
    </ul>
</li>
</ul>



### RIMES

#### Details

RIMES corresponds to french grayscale handwriting images.
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 9,947 |     1,333  | 778 |
| paragraph |  1400 |     100    |  100 |

#### Download
<ul>
<li>Fill in the a2ia user agreement form available [here](http://www.a2ialab.com/doku.php?id=rimes_database:start) and send it by email to rimesnda@a2ia.com. You will receive by mail a username and a password</li>
<li>Login in and download the data from [here](http://www.a2ialab.com/doku.php?id=rimes_database:data:icdar2011:line:icdar2011competitionline)</li>
<li>Move the following files into the folder Datasets/raw/RIMES/
    <ul>
        <li>eval_2011_annotated.xml</li>
        <li>eval_2011_gray.tar</li>
        <li>training_2011_gray.tar</li>
        <li>training_2011.xml</li>
    </ul>
</li>
</ul>


### READ 2016

#### Details
READ 2016 corresponds to Early Modern German RGB handwriting images.
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 8,349 |  1,040    | 1,138|
| paragraph |  1584 |     179    | 197 |

#### Download

<ul>
<li>From root folder:</li>
</ul>

```
cd Datasets/raw
mkdir READ_2016
cd READ_2016
wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
```

### ScribbleLens

#### Details
ScribbleLens corresponds to Early Modern Deutch RGB handwriting images.
The dataset is split as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 4,302 |  481    | 563|

#### Download

<ul>
<li>From root folder:</li>
</ul>

```
cd Datasets/raw
mkdir ScribbleLens
cd ScribbleLens
wget http://openslr.magicdatatech.com/resources/84/scribblelens.{supplement.original.pages.tgz,corpus.v1.2.zip}

```


### Format the datasets

<ul>
<li> Comment/Uncomment the following lines from the main function of the script "format_datasets.py" according to your needs and run it</li>
</ul>

```
if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

    # format_RIMES_line()
    # format_RIMES_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()

    # format_scribblelens_line()
```

<ul>
<li>This will generate well-formated datasets, usable by the training scripts.</li>
</ul>


## Training and evaluation
You need to have a properly formatted dataset to train a model, please refer to the section [Datasets](#Datasets). 

Two scripts are provided to train respectively line and paragraph level models: OCR/line_OCR/ctc/main_line_ctc.py and OCR/document_OCR/v_attention/main_pg_va.py

Training a model leads to the generation of output files ; they are located in the output folder OCR/line_OCR/ctc/outputs/#TrainingName or OCR/document_OCR/v_attention/outputs/#TrainingName.

The outputs files are split into two subfolders: "checkpoints" and "results". "checkpoints" contains model weights for the last trained epoch and for the epoch giving the best valid CER.
"results" contains tensorboard log for loss and metrics as well as text file for used hyperparameters and results of evaluation.

Training can use apex package for mix-precision and Distributed Data Parallel for usage on multiple GPU.

All hyperparameters are specified and editable in the training scripts (meaning are in comments).

Evaluation is performed just after training ending (training is stopped when the maximum elapsed time is reached or after a maximum number of epoch as specified in the training script)

## Citation