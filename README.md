# SegSmall

SegSmall is a project that aims to reduce the computation cost of semantic segmentation models, allowing them to run efficiently on low-power GPU devices. By optimizing the model architecture and training process, SegSmall is able to achieve high-quality segmentation results while significantly reducing the computational resources required. This makes the technique viable for deployment on edge devices, such as mobile phones or embedded systems, where power and resource constraints are important considerations.

## Setup

To use the implementation, you would need to use a virtual environment and install the necessary packages.
We recommend using Anaconda for this:

```bash
conda create -n <environment_name> python=3.11
conda activate <environment_name>
pip install -r requirements.txt
```

Make sure to use Python 3.11 when creating the virtual environment. This version of Python is required for the project to function correctly.
Once you've set up the environment, you should be ready to start using the SegSmall implementation.

## Train

Originally the network has been trained using Cityscapes, if you want to use the same way, follow the next steps:

```
cd dataset
chmod +x download_cityscapes.sh && ./download_cityscapes.sh
```

It will download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds.

If you want more info about it, you can find it in a file called "*readme*" within a directory called *"train"*.

## Use

After each training process, the best weights are saved in a file called according to your training name which ends in ".pth" located inside a folder called results in the train folder.

Therefore, if you want to use the model to make predictions, you only need to take that weights.

## Contact

Feel free to contact with me, if you have any questions, suggeriments or comments about the project:

* Ivan Arroyo (ivan.arpo96@protonmail.com)
