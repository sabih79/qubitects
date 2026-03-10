import tkinter as tk
from tkinter import messagebox, filedialog as fd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

import torchaudio
import soundfile as sf
import numpy as np

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="No gradient function provided*")

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# WAV2VEC (FROZEN)
# =========================================================
bundle = torchaudio.pipelines.WAV2VEC2_BASE
wav2vec = bundle.get_model().to(device)
wav2vec.eval()
for p in wav2vec.parameters():
    p.requires_grad = False

# =========================================================
# CONSTANTS
# =========================================================
IMAGE_SIZE = 224
FEATURE_DIM = 8
APP_W, APP_H = 640, 480

torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# QNN HELPERS
# =========================================================
def z_observables(n):
    obs = []
    for i in range(n):
        s = ["I"] * n
        s[n - 1 - i] = "Z"
        obs.append(SparsePauliOp("".join(s)))
    return obs

def make_qnn(feature_dim):
    fm = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
    ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)

    qc = QuantumCircuit(feature_dim)
    qc.compose(fm, inplace=True)
    qc.compose(ansatz, inplace=True)

    return EstimatorQNN(
        circuit=qc,
        input_params=fm.parameters,
        weight_params=ansatz.parameters,
        observables=z_observables(feature_dim),
    )

# =========================================================
# AUDIO HELPERS
# =========================================================
def load_audio_safe(path):
    audio, sr = sf.read(path, always_2d=True)
    audio = torch.from_numpy(audio.T).float()
    return audio, sr

def to_mono_16k(audio, sr, target_sr=16000):
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio

def load_sound_as_spectrogram(path):
    audio, sr = load_audio_safe(path)
    audio = to_mono_16k(audio, sr)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=128
    )(audio)
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    mel = F.interpolate(
        mel.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )
    return mel

# =========================================================
# MODELS
# =========================================================
class HybridEndToEnd_lungsound(nn.Module):
    def __init__(self, qnn, feature_dim):
        super().__init__()
        self.cnn = models.vgg19(pretrained=True)
        self.cnn.classifier = nn.Identity()

        self.reduce = nn.Sequential(
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

        self.qnn = TorchConnector(qnn)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.reduce(x)
        x = self.qnn(x)
        return torch.sigmoid(self.fc(x))


class HybridEndToEnd_breastultrasound(HybridEndToEnd_lungsound):
    pass


class HybridEndToEnd_breastthermography(nn.Module):
    def __init__(self, qnn, feature_dim):
        super().__init__()
        self.cnn = models.vgg19(pretrained=True)
        self.cnn.classifier = nn.Identity()

        self.reduce = nn.Sequential(
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

        self.qnn = TorchConnector(qnn)
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.reduce(x)
        x = self.qnn(x)
        return self.fc(x)


class HybridWav2VecHQCNN(nn.Module):
    def __init__(self, wav2vec, qnn, feature_dim, num_classes):
        super().__init__()
        self.wav2vec = wav2vec

        self.reduce = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

        self.qnn = TorchConnector(qnn)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, wav):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        with torch.no_grad():
            feats, _ = self.wav2vec(wav)

        x = feats.mean(dim=1)
        x = self.reduce(x)
        x = self.qnn(x)

        if x.shape[1] != FEATURE_DIM:
            raise RuntimeError(f"QNN output mismatch: {x.shape}")

        return self.fc(x)

# =========================================================
# MODEL LOADERS (GLOBAL ALIAS FIX)
# =========================================================
_model_cache = {}

def _load_any(model_file):
    return torch.load(model_file, map_location=device, weights_only=False)

def load_lungsound_model(model_file):
    global HybridEndToEnd
    key = ("lungsound", model_file)
    if key in _model_cache:
        return _model_cache[key]

    HybridEndToEnd = HybridEndToEnd_lungsound
    qnn = make_qnn(FEATURE_DIM)

    obj = _load_any(model_file)

    if isinstance(obj, nn.Module):
        model = obj.to(device)
    elif isinstance(obj, dict):
        model = HybridEndToEnd(qnn, FEATURE_DIM).to(device)
        model.load_state_dict(obj["model_state_dict"])
    else:
        raise RuntimeError("Unsupported lung model format")

    model.eval()
    _model_cache[key] = model
    return model

def load_breastultrasound_model(model_file):
    global HybridEndToEnd
    key = ("ultra", model_file)
    if key in _model_cache:
        return _model_cache[key]

    HybridEndToEnd = HybridEndToEnd_breastultrasound
    qnn = make_qnn(FEATURE_DIM)

    obj = _load_any(model_file)

    if isinstance(obj, nn.Module):
        model = obj.to(device)
    elif isinstance(obj, dict):
        model = HybridEndToEnd(qnn, FEATURE_DIM).to(device)
        model.load_state_dict(obj["model_state_dict"])
    else:
        raise RuntimeError("Unsupported ultrasound model format")

    model.eval()
    _model_cache[key] = model
    return model

def load_breastthermography_model(model_file):
    global HybridEndToEnd
    key = ("thermo", model_file)
    if key in _model_cache:
        return _model_cache[key]

    HybridEndToEnd = HybridEndToEnd_breastthermography
    qnn = make_qnn(FEATURE_DIM)

    obj = _load_any(model_file)

    if isinstance(obj, nn.Module):
        model = obj.to(device)
    elif isinstance(obj, dict):
        model = HybridEndToEnd(qnn, FEATURE_DIM).to(device)
        model.load_state_dict(obj["model_state_dict"])
    else:
        raise RuntimeError("Unsupported thermography model format")

    model.eval()
    _model_cache[key] = model
    return model

def load_heartsound_model(model_file):
    key = ("heart", model_file)
    if key in _model_cache:
        return _model_cache[key]

    qnn = make_qnn(FEATURE_DIM)
    checkpoint = _load_any(model_file)

    model = HybridWav2VecHQCNN(
        wav2vec=wav2vec,
        qnn=qnn,
        feature_dim=FEATURE_DIM,
        num_classes=5
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model_cache[key] = model
    return model

# =========================================================
# INFERENCE
# =========================================================
def check_heartsound(path, wav):
    classes = [
        "Mitral valve prolapse", "Normal",
        "aortic stenosis", "mitral stenosis",
        "mitral regurgitation"
    ]
    audio, sr = load_audio_safe(wav)
    audio = to_mono_16k(audio, sr).to(device)

    model = load_heartsound_model(path)
    with torch.no_grad():
        logits = model(audio)
        probs = torch.softmax(logits, dim=1)
        return classes[probs.argmax(1).item()]

def check_lungsound(path, wav):
    spec = load_sound_as_spectrogram(wav).repeat(1, 3, 1, 1).to(device)
    model = load_lungsound_model(path)
    with torch.no_grad():
        return f"prob={model(spec).item():.4f}"

def check_image(path, loader_fn):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])
    img = tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    model = loader_fn
    with torch.no_grad():
        p = model(img).item()
    return "malignant" if p > 0.5 else "non_malignant"

# =========================================================
# TKINTER UI
# =========================================================
def on_click(label):
    try:
        if label == "Heart Disease":
            f = fd.askopenfilename(filetypes=[("WAV", "*.wav")])
            if f:
                messagebox.showinfo("Result", check_heartsound("heartsound_wav2vec_qnn.pt", f))

        elif label == "Lung Disease":
            f = fd.askopenfilename(filetypes=[("WAV", "*.wav")])
            #if f:
            #    messagebox.showinfo("Result", check_lungsound("Lung_Sound_vgg.pt", f))

        elif label == "Breast Cancer Ultrasound":
            f = fd.askopenfilename(filetypes=[("PNG", "*.png")])
            if f:
                m = load_breastultrasound_model("breastcancer_vgg.pt")
                messagebox.showinfo("Result", check_image(f, m))

        elif label == "Breast Cancer Thermal Image":
            f = fd.askopenfilename(filetypes=[("JPG", "*.jpg")])
            if f:
                m = load_breastthermography_model("breast-thermography_vgg_qnn.pt")
                messagebox.showinfo("Result", check_image(f, m))

        else:
            messagebox.showinfo("Info", "Not implemented")

    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Qubitects: Quantum-Sehat")
root.geometry(f"{APP_W}x{APP_H}")
root.resizable(False, False)

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True)

for t in [
    "Breast Cancer Thermal Image",
    "Breast Cancer Ultrasound",
    "Lung Disease",
    "Heart Disease",
    "Skin Disease"
]:
    tk.Button(frame, text=t, width=30, height=2,
              command=lambda x=t: on_click(x)).pack(pady=8)

root.mainloop()
