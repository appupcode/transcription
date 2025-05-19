
# === TRANSCRIPTION MULTIFICHIER PAR DATE (AVEC REPRISE INTELLIGENTE, LOG FLUSH, .DONE.TXT) ===
# Script pour transcrire tous les fichiers audio d'un dossier en français.
# Découpe les fichiers en morceaux de 5 minutes, traite en parallèle, écrit les résultats progressivement.

import os
import datetime
import time
from math import ceil
from pydub import AudioSegment
import whisper
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support
import torch
import glob

# === CONFIGURATION ===
base_dir = r"G:\scripts\projet311\translator"
audio_dir = os.path.join(base_dir, "audios")
chunks_dir = os.path.join(base_dir, "chunks")
output_dir = os.path.join(base_dir, "transcriptions")

chunk_duration_min = 1
language = "fr"
model_size = "small"
max_workers = max(1, cpu_count() - 1)
supprimer_chunks_apres = False

def decouper_chunk(args):
    i, audio_path, chunk_length_ms, total_length_ms, chunks_dir = args
    audio = AudioSegment.from_file(audio_path)
    start = i * chunk_length_ms
    end = min((i + 1) * chunk_length_ms, total_length_ms)
    chunk = audio[start:end]
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    chunk_path = os.path.join(chunks_dir, f"{audio_name}_chunk_{i}.wav")
    chunk.export(chunk_path, format="wav")
    return chunk_path

def lire_fichiers_termines(log_file):
    if not os.path.exists(log_file):
        return set()
    with open(log_file, "r", encoding="utf-8") as f:
        return set(
            line.split("|")[0].replace("[FICHIER]", "").strip()
            for line in f if line.startswith("[FICHIER]")
        )

def trouver_dernier_fichier(prefixe):
    fichiers = sorted(
        glob.glob(os.path.join(output_dir, f"{prefixe}_*.txt")),
        key=os.path.getmtime,
        reverse=True
    )
    return fichiers[0] if fichiers else None

def main():
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("🧠 Chargement du modèle Whisper (GPU)...")
    print(f"🚀 CUDA dispo : {'Oui' if torch.cuda.is_available() else 'Non'}")
    model = whisper.load_model(model_size, device="cuda")

    existing_chunks = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".wav")])
    if existing_chunks:
        output_file = trouver_dernier_fichier("transcription_globale") or os.path.join(output_dir, "transcription_globale_auto.txt")
        log_file = trouver_dernier_fichier("log") or os.path.join(output_dir, "log_auto.txt")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        output_file = os.path.join(output_dir, f"transcription_globale_{timestamp}.txt")
        log_file = os.path.join(output_dir, f"log_{timestamp}.txt")

    if existing_chunks:
        print(f"🔁 Reprise des {len(existing_chunks)} chunks restants...")
        with open(output_file, "a", encoding="utf-8") as f_out, open(log_file, "a", encoding="utf-8") as log:
            for chunk_file in existing_chunks:
                parts = chunk_file.replace(".wav", "").rsplit("_chunk_", 1)
                if len(parts) != 2:
                    continue
                base_name, i = parts
                chunk_path = os.path.join(chunks_dir, chunk_file)
                done_flag = os.path.join(chunks_dir, f"{base_name}_chunk_{i}.done.txt")
                if os.path.exists(done_flag):
                    continue
                print(f"🎙️ Transcription du chunk {i} en cours...")
                result = model.transcribe(chunk_path, language=language)
                print(f"✅ Chunk {i} terminé")
                with open(done_flag, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                f_out.write(result["text"].strip() + "\n\n")
                f_out.flush()
                log.write(f"Chunk {i} transcrit et sauvegardé.\n")
                log.flush()
                if supprimer_chunks_apres:
                    os.remove(chunk_path)

    fichiers_deja_traite = lire_fichiers_termines(log_file)

    supported_ext = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")
    audios = [
        (f, os.path.getmtime(os.path.join(audio_dir, f)))
        for f in os.listdir(audio_dir)
        if f.lower().endswith(supported_ext) and f not in fichiers_deja_traite
    ]
    audios.sort(key=lambda x: x[1])

    with open(output_file, "a", encoding="utf-8") as f_out, open(log_file, "a", encoding="utf-8") as log:
        total_start = time.time()

        for filename, mod_time in audios:
            input_path = os.path.join(audio_dir, filename)
            mod_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")

            print(f"\n📁 Traitement de : {filename} ({mod_date})")
            f_out.write(f"\n==============================\n")
            f_out.write(f"🎧 {filename}  |  Modifié : {mod_date}\n")
            f_out.write(f"==============================\n\n")
            log.write(f"[FICHIER] {filename} | Date : {mod_date}\n")
            log.flush()

            audio = AudioSegment.from_file(input_path)
            total_length_ms = len(audio)
            chunk_length_ms = chunk_duration_min * 60 * 1000
            num_chunks = ceil(total_length_ms / chunk_length_ms)

            task_args = [
                (i, input_path, chunk_length_ms, total_length_ms, chunks_dir)
                for i in range(num_chunks)
            ]

            print(f"🧩 Découpage en {num_chunks} segments (multi-cœur : {max_workers} workers)...")
            start_cut = time.time()
            with Pool(processes=max_workers) as pool:
                chunk_paths = list(tqdm(pool.imap(decouper_chunk, task_args), total=num_chunks))
            end_cut = time.time()
            cut_time = round(end_cut - start_cut, 2)
            print(f"✅ Découpage terminé en {cut_time} sec")
            log.write(f"Découpage : {cut_time} secondes pour {num_chunks} chunks\n")
            log.flush()

            start_trans = time.time()
            for chunk_path in chunk_paths:
                i = chunk_path.rsplit("_chunk_", 1)[-1].replace(".wav", "")
                base_name = os.path.splitext(os.path.basename(chunk_path))[0].rsplit("_chunk_", 1)[0]
                done_flag = os.path.join(chunks_dir, f"{base_name}_chunk_{i}.done.txt")
                if os.path.exists(done_flag):
                    continue
                print(f"🎙️ Transcription du chunk {i}/{num_chunks - 1}...")
                result = model.transcribe(chunk_path, language=language)
                print(f"✅ Chunk {i} terminé")
                with open(done_flag, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                f_out.write(result["text"].strip() + "\n\n")
                f_out.flush()
                log.write(f"Chunk {i} transcrit et sauvegardé.\n")
                log.flush()
                if supprimer_chunks_apres:
                    os.remove(chunk_path)
            end_trans = time.time()
            trans_time = round(end_trans - start_trans, 2)
            print(f"✅ Transcription terminée en {trans_time} sec")
            log.write(f"Transcription : {trans_time} secondes\n\n")
            log.flush()

        total_end = time.time()
        total_time = round(total_end - total_start, 2)
        log.write(f"Durée totale du traitement : {total_time} secondes\n")
        log.flush()

    print(f"\n✅ Transcription complète enregistrée dans : {output_file}")
    print(f"🗒️  Log sauvegardé dans : {log_file}")

if __name__ == "__main__":
    freeze_support()
    main()
