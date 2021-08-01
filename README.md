# SuperAtomicCharge
Out-of-the-box Deep Learning Prediction of Atomic Partial Charges by Graph Representation and Transfer Learning.
This source code was tested on the basic environment with `conda==4.5.4` and `cuda==11.0`

![Image text](https://github.com/zjujdj/SuperAtomicCharge/blob/main/fig/graph_Abstract.jpg)
## Conda Environment Reproduce
Two methods were provided for reproducing the conda environment used in this paper
- **create environment using file packaged by conda-pack**
    
    Download the packaged file [dgl430_v1_minimum_.tar.gz](https://drive.google.com/file/d/10U4g53LDQSpbGllSi7FomYuFLexFkKn2/view?usp=sharing) 
    and following commands can be used:
    ```python
    mkdir /opt/conda_env/dgl430_v1_minimum
    tar -xvf dgl430_v1_minimum_.tar.gz -C /opt/conda_env/dgl430_v1_minimum
    source activate /opt/conda_env/dgl430_v1_minimum
    conda-unpack
    ```
  
- **create environment using files provided in `./envs` directory**
    
    The following commands can be used:
    ```python
    conda create --prefix=/opt/conda_env/dgl430_v1_minimum --file conda_packages.txt
    source activate /opt/conda_env/dgl430_v1_minimum
    pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -r pip_packages.txt

    ```
  
## Usage
Users can directly use our well-trained model (depoisted in `./model_save/` directory) to predicted the corresponding 
partial atomic charge (DDEC4, DDEC78 and RESP). Because this method was based on the 3D molecular structures. Therefore, 
the actual use of this method should be conducted  on optimized molecules, such as molecules optimized by MMFF force 
field and PM7 method, and the inputs should  be a sdf file containing multiple molecules with 3D coordinates. The input 
example was deposited in `./inputs/3cl-min.sdf` or `./inputs/casp8-min.sdf`. For users who want to train their own model 
using new datasets, we also show a model training example. The corresponding training data was deposited in 
`./training_data`. The label of training data can be assessed using script `./scripts/get_sdf_charge.py`
- **step 1: Clone the Repository**
```python
git clone https://github.com/zjujdj/SuperAtomicCharge.git
```

- **step 2: Construction of Conda Environment**
```python
# method1 in 'Conda Environment Reproduce' section
mkdir /opt/conda_env/dgl430_v1_minimum
tar -xvf dgl430_v1_minimum_.tar.gz -C /opt/conda_env/dgl430_v1_minimum
source activate /opt/conda_env/dgl430_v1_minimum
conda-unpack

# method2 in 'Conda Environment Reproduce' section
cd ./SuperAtomicCharge/envs
conda create --prefix=/opt/conda_env/dgl430_v1_minimum --file conda_packages.txt
source activate /opt/conda_env/dgl430_v1_minimum
pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r pip_packages.txt
```

- **step 3: Charge Prediction**
```python
cd ./SuperAtomicCharge/scripts
nohup python3 -u model_prediction_linux.py --job_of_name=hello_charge --type_of_charge=e4 --input_file=3cl-min.sdf 
--correct_charge --device=cpu > ../outputs/prediction.log 2>&1 &

# model_prediction_linux.py use help
python3 model_prediction_linux.py -h
```

- **step 4: Model Training Example**
```python
cd ./SuperAtomicCharge/scripts
nohup python3 -u model_train.py --gpuid=0 --lr=0.0001 --epochs=5 --batch_size=20 --tolerance=0 --patience=3 --l2=0.000001 
--repetitions=2 --type_of_charge=e4 --num_process=4 --bin_data_file=data_e4.bin > ../outputs/training.log 2>&1 &

# model_prediction_linux.py use help
python model_train.py -h
```

## Acknowledgement
some scripts were based on the [dgl project](https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/attentivefp.py). 
We'd like to show our sincere thanks to them.