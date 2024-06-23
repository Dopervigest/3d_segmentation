# 3D Segmentation Project
[![Python 3.7+](https://img.shields.io/badge/python-3.7_%7C_3.8_%7C_3.9_%7C_3.10_%7C_3.11-blue?style=for-the-badge)](https://www.python.org/downloads/release/python-3110//)

Welcome to my 3d segmentation project page! It is aimed to find the most optimal way to do segmentation of 3d objects. 
The main idea here is to use each point of a 3d object as a single data sample in orded to train the neural network. This increases train and test datasets by a lot and allows training bigger models. You can find out more about using this principle [here](https://ieeexplore.ieee.org/document/10158888).


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project uses basic PyTorch, NumPy and trimesh to implement most important functions, so in order to use it you just need to:
1. Clone the repository: `git clone https://github.com/Dopervigest/3d_segmentation`
2. Navigate to the project directory: `cd 3d_segmentation`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

Just run `jupyter notebook` and use the included .ipynb files. Each of them has a detailed explanation of their purpose inside. 

## Contributing

Thank you for considering contributing to this project!
The best contribution for this project is feedback, please feel free to leave comments in [issues](https://github.com/Dopervigest/3d_segmentation/issues). 

If you want to add some features and/or examples to the existing ones, please consider doing the following:

1. Fork the repository.
2. Create a new branch: `git checkout -b your-feature`
3. Make your changes.
4. Commit your changes: `git commit -m 'Add some feature'`
5. Push to the branch: `git push origin your-feature`
6. Create a new Pull Request.

## License
This project uses MIT Licence.