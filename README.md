# Music Genre Classification - PyTorch
Music Genre classification using Convolutional Neural Networks on Spectrograms in PyTorch

<center><img src = "http://tommymullaney.com/img/cnn-diagram.png"></center>

[Image source](http://tommymullaney.com/img/cnn-diagram.png)

## Using the repository

Clone the repository to the machine where you want to run the model.

```bash
git clone https://github.com/adityashrm21/music-classification-cnn-pytorch.git
```
It is preferable but not necessary to use a GPU.

### Preparing the dataset

Go to the root directory of the cloned repository and run the following commands:

```bash
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -xvzf genres.tar.gz
```

### Running the model

You can train the model and do the inference on test set using:

```bash
python3 train.py --root_dir "." --lr 1e-3 --momentum 0.9 --epochs 50
```
Here is a complete list of arguments:

```bash
usage: train.py [-h] --root_dir ROOT_DIR [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--lr LR] [--momentum MOMENTUM]
                [--weight_decay WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   root directory for the dataset
  --epochs EPOCHS       num of training epochs
  --batch_size BATCH_SIZE
                        training batch size
  --lr LR               learning rate
  --momentum MOMENTUM   momentum for SGD
  --weight_decay WEIGHT_DECAY
                        weight decay for L2 penalty
```

### Results

Current test accuracy with **100 epochs: 75%**.

**Todo**:

- Try pretrained model architectures
