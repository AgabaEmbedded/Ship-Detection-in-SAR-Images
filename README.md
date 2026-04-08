
---

# Ship-Detection-in-SAR-Images

### **Overview**
This project implements a high-precision object detection pipeline specifically optimized for **Synthetic Aperture Radar (SAR)** imagery. Detecting maritime vessels in SAR data presents unique challenges, including high speckle noise, extreme background-to-target imbalance, and wide dynamic ranges. 

This system utilizes the **YOLO26** architecture, featuring an NMS-free end-to-end inference engine, tailored for deployment in resource-constrained environments.



## **Project Workflows**

### **Training Pipeline**

The training is conducted using a Distributed Data Parallel (DDP) setup, optimized for multi-GPU environments (Dual Tesla T4).

  * [**View Training Notebook**](https://github.com/AgabaEmbedded/Ship-Detection-in-SAR-Images/blob/main/src/Training-Notebook.ipynb) — *Detailed breakdown of the YOLO26 training process, hyperparameter tuning, and optimization.*

### **Inference & Deployment**

The inference pipeline handles raw SAR uploads by automatically applying the required contrast stretching before patching and prediction.

  * [**View Inference Notebook**](https://github.com/AgabaEmbedded/Ship-Detection-in-SAR-Images/blob/main/src/Inference-Notebook.ipynb) — *Demonstration of the end-to-end detection pipeline on unseen SAR scenes.*

-----

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.