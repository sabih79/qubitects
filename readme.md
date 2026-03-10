Qubitects
---------

Hybrid Quantum-Classical AI for Medical Diagnostics

Qubitects is a research prototype demonstrating hybrid quantum-classical machine learning models for medical diagnostics using deep learning and quantum neural networks.

The project explores the integration of classical neural networks with quantum machine learning layers, enabling experimental workflows that combine PyTorch deep learning with quantum circuits.

Overview
--------

Recent advances in Quantum Machine Learning (QML) aim to combine classical machine learning with quantum computing techniques to improve learning efficiency and scalability.

Qubitects investigates hybrid models for medical signal and image analysis, including:

	Lung sound classification
	Heart sound analysis
	Breast cancer detection
	Breast thermography diagnostics
	Skin disease classification

The system combines classical CNN architectures with Quantum Neural Network (QNN) layers, enabling hybrid learning pipelines.

Architecture
------------

The hybrid architecture typically follows this structure:

Input Data -> Feature Extractor (CNN / VGG / Wav2Vec) -> Classical Dense Layers -> Quantum Neural Network (Variational Quantum Circuit) -> Classifier Output

The project uses Qiskit Machine Learning integrated with PyTorch through the TorchConnector, enabling hybrid quantum-classical model training.

Repository Structure
--------------------

## Repository Structure

```
qubitects/
│
├── qubitects.py
│
├── training/
│   ├── hybrid_vgg_qnn_torchconnector_breast_ultrasound.ipynb
│   ├── hybrid_vgg_qnn_torchconnector_breast_thermography.ipynb
│   ├── hybrid_vgg_qnn_torchconnector_lungsound3.ipynb
│   ├── hybrid_vgg_qnn_torchconnector_heartsound2.ipynb
│   ├── hybrid_vgg_qnn_torchconnector_skin_disease.ipynb
│
├── Lung_Sound_vgg.pt
├── breastcancer_vgg.pt
├── breast-thermography_vgg_qnn.pt
├── heartsound_wav2vec_qnn.pt
│
└── README.md
```

Models Included
| Model                            | Description                                    |
| -------------------------------- | ---------------------------------------------- |
| `Lung_Sound_vgg.pt`              | Lung sound classification model                |
| `breastcancer_vgg.pt`            | Breast cancer ultrasound detection             |
| `breast-thermography_vgg_qnn.pt` | Breast thermography hybrid QNN                 |
| `heartsound_wav2vec_qnn.pt`      | Heart sound classification using Wav2Vec + QNN |


Installation
------------

Clone the repository:

git clone https://github.com/sabih79/qubitects.git
cd qubitects

Install dependencies:

pip install torch
pip install qiskit
pip install qiskit-machine-learning
pip install numpy matplotlib scikit-learn
Training the Models

To train models, run the relevant Jupyter Notebook file inside the training folder.

Examples:
---------

jupyter notebook training/hybrid_vgg_qnn_torchconnector_breast_ultrasound.ipynb
jupyter notebook training/hybrid_vgg_qnn_torchconnector_breast_thermography.ipynb
jupyter notebook training/hybrid_vgg_qnn_torchconnector_lungsound3.ipynb
jupyter notebook training/hybrid_vgg_qnn_torchconnector_heartsound2.ipynb
jupyter notebook training/hybrid_vgg_qnn_torchconnector_skin_disease.ipynb

Each notebook contains the complete training pipeline including data preprocessing, model architecture, training loop, and evaluation metrics.

Running the GUI
---------------

To launch the graphical user interface:

python qubitects.py

The GUI allows users to load trained models and perform AI-assisted medical diagnostics.

Applications
------------

Potential applications include:

	Low-cost medical diagnostics
	AI-assisted rural healthcare
	Edge-AI medical devices
	Hybrid quantum-classical machine learning research

Future Work
-----------

Planned extensions include:
	Quantum Convolutional Neural Networks
	Quantum Feature Maps for medical imaging
	Deployment on real quantum hardware
	Integration with portable diagnostic devices

Authors
-------

Team Qubitects (Students: Asim Javed, Fatima Zafar, Mahum Noor, Muneeb ur Rehman, Saif ullah, Syeda Afia Shah) (Mentors: Dr. Abdul Razzaque and Dr. Sabih D. Khan)

License
-------

This project is released for research and educational purposes.
