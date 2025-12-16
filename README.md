# CS374FINAL

the file structure looks like this:
```
/project
  /finalform
  /midterm
```
ALL THE DATA MANIPULATION FILES WILL PROBABLY NOT WORK BECAUSE THE DATASETS WERE TO BIG TO UPLOAD.

Anything in finalform was done after the midterm so this is implementing with the dinoV2 encoder and grabbing more data.
For this section the data folder contains sample data that can be used to test some of the files.

Here is how the project is intended to run:

What is given in the data:
/MergedDataset - merged_master_tabular_0p1 - I have provided you with a pre merged data set that contains information on the optical image location and tabular information for a given position in space.
unique_master_indicies.pkl - This is intended to keep track of galaxy position in relation to a global key called ("master_index") that is used in other tables like in the merged data set
dinov2_embeddings.npy - These images are already pre embedded but feel free to run it again but be aware that when you run the script this will be over written
dinov2_index.csv - this is also outputted when you run the dino embed script

For this section there are two possible routes, the version where data is not leaked in the autoencoder while training and the version where it is.

I will walk through both approaches:

 create virtual environment and download dependencies:
  (for mac)
  ```bash
  cd FinalForm
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
1. non-leak
  Then to run everything do:
  ```bash
  python train_flux_models_leakfree.py
   ```
  you will see graphs and other data appear in the data folder

2. leak
  First embed the tabular data do:
  ```bash
  python embed_tabular.py
  ```
  then run:
  ```bash
  python train_flux_models.py
  ```
  Finally for graphs run:
  ```bash
  python plot_test_graphs_reportcard.py
  ```
Anything in midterm is from the midterm report so that includes using resnet and the autoencoder.

```bash
cd midterm
python -m venv .venv
source .venv/bin/activate   # mac/linux
pip install -r midterm_requirements.txt
```

Run these programs for results:

```bash
python make_midterm_sample_data.py
python vectorize_data.py
python reembed_tabular.py
python train_flux_models.py
```
