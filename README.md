# stylegan-detector

A reimplementation of [What Makes Fake Images Detectable? Understanding Properties that Generalize](https://arxiv.org/pdf/2008.10588.pdf)

By [Nam Do](https://ndo3.github.io) '21, Julia Windham '21, and Esmeralda Montas '21 - Brown University

### Overview

We replicated the experimental design of this paper, but with a few caveats:

- The data that we use is from this [Dataset of 140,000 Images with a binary label of real/fake](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)

- We follow exactly the image pre-processing design choices of the aforementioned paper.

### Folder structure & Running instructions

We used the following:
- Python version: 3.8.5

The data is not pushed onto the repository. In order to replicate our code, do the following:

1. Follow the link to the aforementioned dataset, and unzip them to `data` such that the folders `train`, `test`, and `valid` are in `data` (e.g. `/data/train`, `/data/test`, `data/valid`)

### Data