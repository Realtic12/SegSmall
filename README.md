# SegSmall

SegSmall is a project that aims to reduce the computation cost of semantic segmentation models, allowing them to run efficiently on low-power GPU devices. By optimizing the model architecture and training process, SegSmall is able to achieve high-quality segmentation results while significantly reducing the computational resources required. This makes the technique viable for deployment on edge devices, such as mobile phones or embedded systems, where power and resource constraints are important considerations.

## Structure

| Layer | Type                            | Feature Map Number | Output Resolution | Notes                                   |
|-------|---------------------------------|--------------------|-------------------|-----------------------------------------|
| 1     | Initial Block                   | 16                 | 512 x 256         | Conv, Max Pooling                       |
| 2-6   | Downsampler Block               | 64                 | 256 x 128         | Conv, Max Pooling                       |
| 7     | Non-bt-1d (dilation 2)          | 64                 | 256 x 128         | Depthwise separable convolutions, SE Block |
| SE    | SE Block                        | 64                 | 256 x 128         | Squeeze-and-Excitation                  |
| 8     | Downsampler Block               | 128                | 128 x 64          | Conv, Max Pooling                       |
| 9     | Non-bt-1d (dilation 2)          | 128                | 128 x 64          | Depthwise separable convolutions        |
| 10    | Non-bt-1d (dilation 4)          | 128                | 128 x 64          | Depthwise separable convolutions        |
| 11    | Non-bt-1d (dilation 8)          | 128                | 128 x 64          | Depthwise separable convolutions        |
| 12    | Non-bt-1d (dilation 16)         | 128                | 128 x 64          | Depthwise separable convolutions        |
| 13    | Non-bt-1d (dilation 2)          | 128                | 128 x 64          | Depthwise separable convolutions        |
| 14    | Non-bt-1d (dilation 4)          | 128                | 128 x 64          | Depthwise separable convolutions        |
| 15    | Non-bt-1d (dilation 8)          | 128                | 128 x 64          | Depthwise separable convolutions        |
| 16    | Non-bt-1d (dilation 16)         | 128                | 128 x 64          | Depthwise separable convolutions        |
| SE2   | SE Block                        | 128                | 128 x 64          | Squeeze-and-Excitation                  |
| 17    | Upsampler Block                 | 64                 | 256 x 128         | Resolution enhancement                  |
| 18-19 | Non-bt-1d                       | 64                 | 256 x 128         | Depthwise separable convolutions        |
| 20    | Upsampler Block                 | 16                 | 512 x 256         | Resolution enhancement                  |
| 21-22 | Non-bt-1d                       | 16                 | 512 x 256         | Depthwise separable convolutions        |
| 23    | Transposed Convolution          | Number of classes  | 1024 x 512        | Final prediction                        |


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

Also if you want to automate the conversion to train labels, you can use:
```
cd dataset
chmod +x process_images.sh && ./process_images.sh
```

Finally, if you want more info about it, you can find it in a file called "*readme*" within a directory called *"train"*.

## Use

After each training process, the best weights are saved in a file called according to your training name which ends in ".pth" located inside a folder called results in the train folder.

Therefore, if you want to use the model to make predictions, you only need to take that weights.

## Contact

Feel free to contact with me, if you have any questions, suggeriments or comments about the project:

* Ivan Arroyo (ivan.arpo96@protonmail.com)
