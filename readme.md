# ACP-DRL
## About
ACP-DRL: An Anticancer Peptides Recognition Method Based on Deep Representation Learning.

## Hardware Support
We ran ACP-DRL on a single node of the GPU cluster in the National Center for Protein Sciences (Beijing). This node is equipped with two 2.6GHz Intel Xeon processors, eight Tesla V100 GPUs, 256 GB RAM and runs under CentOS 7.6.

Based on our practical experience, we recommend running on a GPU with at least 12GB of memory.

## Installation
1. Use the git command or download the zip file locally.
    ~~~
    git clone  https://github.com/shallFun4Learning/ACP-DRL.git
    ~~~
2. Dependency
    Base Dependency:
    >    Python 3.7.11/3.8.3
    >
    >    PyTorch 1.7.1/1.12.0
    >
    >    cudnn 7.6.5
    >
    >    transformers 4.32.1
    >
    >    tokenizers 0.13.2
    >
    >    datasets 2.16.1
    >
    >    scikit-learn 1.3.0
    > 
   We recommend using conda for environment management.
   
   a. To create a new environment.
   ~~~
   conda create -n YOUR_ENV_NAME python=3.8.3
   ~~~
   
   b. Switch to the created environment.
   ~~~
   conda activate YOUR_ENV_NAME
   ~~~
   
   c. Install PyTorch
   ~~~
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
   ~~~
    or see the [PyTorch website](https://pytorch.org/get-started/previous-versions/).
   
   d. About other dependencies

   Now, let's install transformers
   ~~~
   pip install transformers
   ~~~
   and other dependencies.

## Usage
### Weights
The available weights file will be updated our OneDrive.

[OPP](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EY7n6YoZaFpGnV-u_f1N9cgBdWUOrp3TGc8Ko2IhqrI8wg?e=LZtS7t)

[Main](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/ES_FQM3g8h9EvUdYT92lWTABds4XFNW6y4eDlqxzApR-cg?e=PrIOxG)

[Alternate](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/ERIN3xCdLj9BkbxsKQiZOLMBspwxF6z6Ur5xDj50mAXXBA?e=m7Of4I)

and

[Tokenizer config](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EVxWqoQUlvRLmkvBgRg594UBSN2gMqLZeK3OtneA2nnoTg?e=M14WN0)

### Datasets
The available datasets file will be updated our OneDrive.

[Main](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EUCXUBO1CiZJsX94Xe_EfY4BfGV2uUdYw3YAQKhvVT9MPg?e=3RSG3b)

[Alternate](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EWyrkuhxcVpKoT8QHjUtQasBfISiS6PEyuz3V7PZvYrbVA?e=VHHRAy)

[IFPT](https://4wgz12-my.sharepoint.com/:u:/g/personal/admin_4wgz12_onmicrosoft_com/EXnv2TxfvM9GkiDnr87242QBPv7zkgMJDYup9iJ87ZoW-Q?e=ZV95Hx)

### Fasta to CSV Conversion Script
This script converts .fasta files to .csv format. It treats the portion of each line before the "|" as the title, the portion after the "|" as the label, and the next line as the sequence, forming each column of the .csv file.

~~~
python convert_script.py /path/to/input1.fasta /path/to/input2.fasta ...
~~~

Arguments:
`--infiles`: a list of one or more paths to .fasta files that you want to convert. Replace "/path/to/input1.fasta" and "/path/to/input2.fasta" in the sample code above with the actual paths to your files. You can add as many .fasta file paths as you need.

Output:This script generates a .csv file in the same directory as each input file. The new file has the same name as the original .fasta file, but with a different file extension.


### Quick start

~~~
python run.py \
    --model_path YOUR_MODEL_PATH \
    --test_dataset_path YOUR_TEST_SET_PATH\
    --tokenizer_path YOUR_TOKENIZER_PATH\
    --outPutDir YOUR_OUTPUT_PATH
~~~

Explanation for each parameter:

1. `--model_path YOUR_MODEL_PATH` : This is where your model file is stored. Replace 'YOUR_MODEL_PATH' with the full path to your model file. For example, if your model is stored in a directory named "models" with the model file named "model.pth", your model path would be `"users/models"`.

2. `--test_dataset_path YOUR_TEST_SET_PATH` : This is where your test dataset file resides. Replace 'YOUR_TEST_SET_PATH' with the full path to your test dataset. For instance, if your dataset is stored in a directory named "data" with the file named "test_dataset.csv", your testset path would be `"data/test_dataset.csv"`.In this project, the CSV file is required to have at least two columns: sequence and label.

3. `--tokenizer_path YOUR_TOKENIZER_PATH` : This is where your tokenizer configuration file is located. Replace 'YOUR_TOKENIZER_PATH' with the full path to your tokenizer configuration file. For instance, if your tokenizer configuration is stored in a directory named "tokenizer" with the file named "tokenizer_config.json", your tokenizer path would be `"users/tokenizer"`.

4. `--outPutDir YOUR_OUTPUT_PATH` : This is where the results of model execution will be stored. Replace 'YOUR_OUTPUT_PATH' with the full path to your desired output location. For instance, it might look like this: `"output/directory"`.

Please make sure to replace 'YOUR_MODEL_PATH', 'YOUR_TEST_SET_PATH', 'YOUR_TOKENIZER_PATH', and 'YOUR_OUTPUT_PATH' with real paths in your environment.

## LICENSE
ACP-DRL is for non-commercial use only.

## Supports
Feel free to submit an issue or contact the author(sfun@foxmail.com) if you encounter any problems during use.

Happy New Year 2024

 :-)

