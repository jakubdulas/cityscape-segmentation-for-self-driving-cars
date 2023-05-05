# Cityscape Segmentation for Self-Driving Cars

This repository contains code for a semantic segmentation model trained on the Cityscapes dataset. The model is designed to be used for self-driving cars to identify the different objects in urban environments.

## Installation

To install the necessary packages, run the following command:

```
pip install -r requirements.txt
```

## Usage

To use the segmentation model, run the following command:

```
python segment.py <image_path>
```

The model will segment the image and output an image with the different objects in the image highlighted with different colors.

## Model

The segmentation model is a U-Net architecture that was trained on the Cityscapes dataset. The model is able to segment various objects in urban environments, such as roads, buildings, and vehicles.

The trained weights and model can be downloaded from the following links:

- [unet_vgg19_backbone_30_epochs_weights.h5](https://www.dropbox.com/s/ckbb9vt44qok7u7/unet_vgg19_backbone_30_epochs_weights.h5?dl=0)
- [unet_vgg19_backbone_30_epochs.h5](https://www.dropbox.com/s/7nmmo3z65ci2jqg/unet_vgg19_backbone_30_epochs.h5?dl=0)

Download both files and place them in the same directory as `CityscapeSegmentation.ipynb`.

## Dataset

The dataset used to train the model is the Cityscapes dataset. The dataset contains images of urban environments with pixel-level annotations for 30 different object classes.

## Contributing

Contributions to this project are welcome. If you find a bug or have a feature request, please open an issue. If you would like to contribute code, please open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
