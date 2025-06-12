# TeKoto
# TeKoto — ASL to Text Translator

**TeKoto** is an AI-powered app that translates **American Sign Language (ASL)** hand gestures into text using a deep learning model built with **PyTorch**. Inspired by the precision of ASL.

## Features

- **Real-time ASL letter recognition** — _Work in progress_  
- **Custom CNN architecture** — may be dumb now, but it's learning fast  
- **Accuracy & loss tracking** — handled on the backend  
- **Confusion matrix** — for understanding where the model is confused (like us sometimes)

---

## Dataset Structure

This project uses the [**ASL Alphabet Archive**](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
```bash
ASL Alphabet Archive/
└── asl_alphabet_train/
├── A/
│ ├── A1.jpg
│ ├── A2.jpg
├── B/
├── ...
└── Z/
```
Ensure the dataset path is set correctly in your script.

---

## Getting Started

1. **Clone this repo**:

```bash
git clone https://github.com/yourusername/tekoto.git
cd tekoto
``` 
Add the dataset to the project root (see folder structure above)

Train the model:
```bash
python main.py
```
## Example Training Output 
```bash
Epoch 1 — Loss: 3.37, Val Accuracy: 3.4%
Epoch 10 — Loss: 2.43, Val Accuracy: 23.6%
Confusion matrix gets generated after training, highlighting misclassifications.
```
## Built With
* Python 3.10+

* PyTorch

* NumPy

* Matplotlib

* scikit-learn

## Name
TeKoto = 手 (Te: Hand) + 事 (Koto: Thing / Matter)
Random idea at 12 am.

## Contributions Welcome
Pull requests, ideas, and critiques all welcome. Feel free to fork and build!

## Links
Dataset: ASL Alphabet Archive on [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

Author: @dabby12 

## Notes
* this app was built for a hackton... may work on in the future 

## Custom cites
@misc{https://www.kaggle.com/grassknoted/aslalphabet_akash nagaraj_2018,
title={ASL Alphabet},
url={https://www.kaggle.com/dsv/29550},
DOI={10.34740/KAGGLE/DSV/29550},