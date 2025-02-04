---
title: "Attention is all you need" 
date: 2021-04-06
lastmod: 2024-10-18
tags: ["social psychology","inner hedgehog","academia","experimental psychology","invasive thoughts"]
author: ["Hilda Schreiber-Ziegler", "Moritz-Maria von Igelfeld"]
description: "This paper describes the inner hedgehog, a psychological condition widespread in academia. Published in the Journal of Socio-Experimental Psychology, 2021." 
summary: "Using several case studies, this paper describes the inner hedgehog, a psychological condition widespread in academic occupations. The condition has lasting consequences and no known cure." 
cover:
    image: "paper3.png"
    alt: "Vizualisation of an inner hedgehog"
    relative: true
editPost:
    URL: "https://github.com/pmichaillat/hugo-website"
    Text: "Journal of Socio-Experimental Psychology"

---

---

##### Download

+ [Paper](paper3.pdf)
+ [Raw data](https://github.com/pmichaillat/recession-indicator)

---

##### Abstract

Using several case studies, this paper describes the inner hedgehog, a psychological condition widespread in academic occupations. The condition has lasting consequences and no known cure. Mauris tincidunt quam a libero consequat, nec pharetra nunc tristique. Pellentesque eget ipsum ut dui laoreet congue ut nec nulla. Nulla facilisi. Sed consequat, odio ac aliquet tempor, turpis augue auctor mauris, at malesuada sem dolor eget libero. Nullam iaculis malesuada risus, id fringilla quam sagittis ac. Fusce congue vel ex et facilisis. Integer volutpat eros ut urna efficitur, id efficitur sapien pharetra.

---

##### Citation

Schreiber-Ziegler, Hilda, and Moritz-Maria von Igelfeld. 2021. "Your Inner Hedgehog." *Journal of Socio-Experimental Psychology* 131 (2): 1299–1302.

```BibTeX
@article{SZI21,
author = {Hilda Schreiber-Ziegler and Moritz-Maria von Igelfeld},
year = {2021},
title ={Your Inner Hedgehog},
journal = {Journal of Socio-Experimental Psychology},
volume = {131},
number = {2},
pages = {1299--1302}}
```

---

# 🚀 Building a CNN Model in PyTorch

## 📌 Introduction  
In this blog, we will implement a **Convolutional Neural Network (CNN)** using PyTorch to classify images. The architecture consists of:
- **Two convolutional layers** with **ReLU activation**
- **Max-pooling layers** for feature extraction
- **A fully connected layer** for classification

---

## 🏗️ **Model Architecture**  
Below is a summary of the CNN model:

| Layer       | Type          | Output Shape  |
|------------|--------------|--------------|
| Input      | `1x28x28`    | 28x28x1      |
| Conv2D     | `3x3, 8 filters`  | 28x28x8   |
| ReLU       | Activation    | 28x28x8      |
| MaxPool    | `2x2`        | 14x14x8      |
| Conv2D     | `3x3, 16 filters` | 14x14x16 |
| ReLU       | Activation    | 14x14x16     |
| MaxPool    | `2x2`        | 7x7x16       |
| Flatten    | -            | 1x784        |
| Fully Connected | `Linear` | `num_classes` |

---

## 🛠 **Implementing CNN in PyTorch**
```python
import torch

# 🏗️ Defining a CNN Model in PyTorch
class PyTorchModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes  # 🎯 Define number of output classes

        # 🧩 Feature Extractor: Convolutional Layers + Activation + Pooling
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 🏗️ Conv Layer 1
            torch.nn.ReLU(),  # ⚡ Activation Function
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 🔽 Downsampling
            
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # 🏗️ Conv Layer 2
            torch.nn.ReLU(),  # ⚡ Activation
            
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 🔽 Downsampling
        )

        # 🏁 Fully Connected Output Layer
        self.output_layer = torch.nn.Linear(7*7*16, num_classes)  # 📊 Final classification layer

    def forward(self, x):
        x = self.features(x)  # 🔄 Forward pass through CNN
        x = torch.flatten(x, start_dim=1)  # 📝 Flatten tensor for FC layer
        x = self.output_layer(x)  # 🎯 Compute logits (raw scores)
        return x
```

##### Related material

+ [Nontechnical summary](https://www.alexandermccallsmith.com/book/your-inner-hedgehog)
