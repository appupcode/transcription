
# 🐍 Environnement virtuel Python 3.11 pour Transcription Whisper

Ce fichier décrit les étapes pour installer et configurer un environnement virtuel Python 3.11 (`venv311`) utilisé pour exécuter le script de transcription audio `TranscriptionGPU.py`.

---

## ✅ 1. Installer Python 3.11

- Télécharger : https://www.python.org/downloads/release/python-3110/
- Cochez **"Add Python to PATH"** pendant l'installation.

---

## 📁 2. Créer un environnement virtuel

```bash
python -m venv G:\scripts\python3.11\venv311
```

---

## ⚙️ 3. Activer l'environnement virtuel

### PowerShell :

```powershell
G:\scripts\python3.11\venv311\Scripts\Activate.ps1
```

### CMD :

```cmd
G:\scripts\python3.11\venv311\Scripts\activate.bat
```

---

## 📦 4. Installer les dépendances

```bash
pip install --upgrade pip setuptools wheel
pip install git+https://github.com/openai/whisper.git
pip install pydub tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎧 5. Installer FFmpeg

- Télécharger depuis : https://ffmpeg.org/download.html
- Extraire et ajouter le dossier `bin/` dans la variable d’environnement `PATH`

### Vérifier :

```bash
ffmpeg -version
```

---

## 📂 6. Structure attendue du projet

```
G:\scripts\projet311\translator\
├── audios\
├── chunks\
├── transcriptions\
└── TranscriptionGPU.py
```

---

## ▶️ 7. Exécuter le script sans activer le venv

```bash
G:\scripts\python3.11\venv311\Scripts\python.exe G:\scripts\projet311\translator\TranscriptionGPU.py
```

---

## 🧠 Remarques

- L’environnement `venv311` est utilisé pour séparer proprement les dépendances de ce projet.
- Le script est optimisé pour un usage GPU mais fonctionne aussi en CPU.
