# NLOS Classification

## Data Prepreation

Follow the steps in `data_preprocess.ipynb` to build the dataset.
The file structure in `project_root` is as follow:
```
project_root
|   README.md
|   train.py
|   data_preprocess.ipynb
|   test.ipynb
+---raw_pngs
|   +---dataset1_name
|   |   +---class1_timestamp1
|   |   |       xxx.mat
|   |   |       1.png
|   |   |       2.png
|   |   |       ...
|   |   +---class1_timestamp2
|   |   |       xxx.mat
|   |   |       1.png
|   |   |       2.png
|   |   |       ...
|   |   ...
|   |   +---class2_timestamp3
|   |   |       xxx.mat
|   |   |       1.png
|   |   |       2.png
|   |   |       ...
|   |   ...
|   +---dataset2_name
|   ...
+---my_utils
    ...
```

## Training

Create a new configuration file in `./configs` or directly use `default.yaml` for training. 
To train a model, run the code below for example:
```
python train.py --config=configs/default.yaml
```

## Testing

Follow the code blocks in `test.ipynb` to test a trained model, 
