import os
import sys
import io
import re
import time
import psutil
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import customtkinter
from PIL import Image

# Suppress Hugging Face symlink warnings globally before importing transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TextRedirector:
    """Keep GUI logs readable and format terminal/HF progress bars nicely in GUI."""

    ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, widget):
        self.widget = widget
        self.last_progress_text = ""

    def write(self, str_val):
        if not str_val:
            return

        # Safe terminal write (prevents NoneType crash in PyInstaller --windowed mode)
        if sys.__stdout__ is not None:
            try:
                sys.__stdout__.write(str_val)
                sys.__stdout__.flush()
            except Exception:
                pass

        # Clean ANSI escape sequences
        cleaned = self.ANSI_ESCAPE.sub('', str_val)

        # Handle carriage returns / progress updates (tqdm / HuggingFace output)
        if "\r" in str_val or "Downloading" in cleaned or "Fetching" in cleaned:
            lines = [l.strip() for l in cleaned.replace('\r', '\n').split('\n') if l.strip()]
            for line in lines:
                if any(k in line for k in ["%", "MB/s", "GB/s", "Downloading", "Fetching", "Reconstructing"]):
                    clean_line = re.sub(r'\[[A-Z]', '', line).strip()
                    if clean_line and clean_line != self.last_progress_text:
                        self.last_progress_text = clean_line
                        self.widget.after(0, self._update_last_line, f"-> {clean_line}")
            return

        clean_text = cleaned.strip()
        if clean_text:
            self.widget.after(0, self._append_text, clean_text)

    def _append_text(self, str_val):
        try:
            if self.widget.winfo_exists():
                self.widget.configure(state="normal")
                self.widget.insert("end", str_val + "\n")
                self.widget.see("end")
                self.widget.configure(state="disabled")
        except Exception:
            pass

    def _update_last_line(self, str_val):
        try:
            if self.widget.winfo_exists():
                self.widget.configure(state="normal")
                self.widget.insert("end", str_val + "\n")
                self.widget.see("end")
                self.widget.configure(state="disabled")
        except Exception:
            pass

    def flush(self):
        if sys.__stdout__ is not None:
            try:
                sys.__stdout__.flush()
            except Exception:
                pass


customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("blue")

BG = "#F8FAFC"
SIDEBAR = "#EDF2F7"
PANEL = "#FFFFFF"
PANEL_ALT = "#F1F5F9"
BORDER = "#CBD5E1"
TEXT = "#0F172A"
MUTED = "#64748B"
ACCENT = "#2563EB"
ACCENT_HOVER = "#1D4ED8"
GREEN = "#10B981"
RED = "#EF4444"

tokenizer = None
model = None
stop_token_ids = []
runtime_device = "cuda"

POPULAR_MODELS = {
    "Mistral-7B-v0.1": {
        "repo": "mistralai/Mistral-7B-v0.1",
        "supported": True
    },
    "Llama-3-8B (Coming Soon)": {
        "repo": "meta-llama/Meta-Llama-3-8B",
        "supported": False
    },
    "Gemma-2-9B (Coming Soon)": {
        "repo": "google/gemma-2-9b",
        "supported": False
    },
    "Phi-3-Mini-4K (Coming Soon)": {
        "repo": "microsoft/Phi-3-mini-4k-instruct",
        "supported": False
    },
    "Qwen-2.5-7B (Coming Soon)": {
        "repo": "Qwen/Qwen2.5-7B",
        "supported": False
    }
}


class ModelSelectionWindow(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("AmpleFlow — Setup")
        self.geometry("720x750")
        self.minsize(680, 550)
        self.configure(fg_color=BG)
        self._center_window(720, 750)

        self.selected_path = os.path.abspath("./huggingface/hub")
        self.download_target_size_bytes = 14.5 * (1024 ** 3)
        self.is_downloading = False

        self._build_ui()
        self._redirect_terminal_output()
        self._check_hardware_requirements()
        
        self.after(500, lambda: self._fetch_model_size_dynamic(self.model_dropdown.get()))

    def _center_window(self, w, h):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _redirect_terminal_output(self):
        redirector = TextRedirector(self.log_box)
        sys.stdout = redirector
        sys.stderr = redirector

    def get_base_dir(self):
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller EXE inside MSIX package
            return os.path.dirname(sys.executable)
        # Running as standard Python script
        return os.path.dirname(os.path.abspath(__file__))

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.scroll_frame = customtkinter.CTkScrollableFrame(
            self, fg_color="transparent", corner_radius=0
        )
        self.scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        header = customtkinter.CTkFrame(self.scroll_frame, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        header.pack(fill="x", padx=10, pady=(10, 10))

        # --- LOGO INTEGRATION (SETUP HEADER) ---
        logo_path = os.path.abspath(os.path.join(self.get_base_dir(), ".Icons", "1.png"))
        if os.path.exists(logo_path):
            header_img = Image.open(logo_path)
            self.header_logo = customtkinter.CTkImage(light_image=header_img, dark_image=header_img, size=(48, 48))
            
            logo_frame = customtkinter.CTkFrame(header, fg_color="transparent")
            logo_frame.pack(anchor="w", padx=20, pady=(16, 12))

            customtkinter.CTkLabel(logo_frame, image=self.header_logo, text="").pack(side="left", padx=(0, 12))

            text_frame = customtkinter.CTkFrame(logo_frame, fg_color="transparent")
            text_frame.pack(side="left")

            customtkinter.CTkLabel(
                text_frame, text="AmpleFlow", font=("Segoe UI", 22, "bold"), text_color=ACCENT
            ).pack(anchor="w")

            customtkinter.CTkLabel(
                text_frame, text="Pocket LLM — Setup & Initialization",
                font=("Segoe UI", 11), text_color=MUTED
            ).pack(anchor="w")
        else:
            customtkinter.CTkLabel(
                header, text="AmpleFlow", font=("Segoe UI", 22, "bold"), text_color=ACCENT
            ).pack(anchor="w", padx=20, pady=(16, 2))

            customtkinter.CTkLabel(
                header, text="Pocket LLM — Setup & Initialization",
                font=("Segoe UI", 11), text_color=MUTED
            ).pack(anchor="w", padx=20, pady=(0, 16))

        config_card = customtkinter.CTkFrame(self.scroll_frame, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        config_card.pack(fill="x", padx=10, pady=6)

        customtkinter.CTkLabel(config_card, text="Select Base LLM Model", font=("Segoe UI", 12, "bold"), text_color=TEXT).pack(anchor="w", padx=20, pady=(14, 4))
        
        all_models = list(POPULAR_MODELS.keys())
        self.current_selected_model = all_models[0]

        self.model_dropdown = customtkinter.CTkOptionMenu(
            config_card, values=all_models,
            height=38, corner_radius=9, fg_color=PANEL_ALT, text_color=TEXT,
            button_color=BORDER, button_hover_color=ACCENT, command=self._on_model_select
        )
        self.model_dropdown.pack(fill="x", padx=20, pady=(0, 6))

        self.model_info_lbl = customtkinter.CTkLabel(
            config_card, text="Fetching model download size from Hugging Face Hub...",
            font=("Segoe UI", 10, "bold"), text_color=ACCENT
        )
        self.model_info_lbl.pack(anchor="w", padx=20, pady=(0, 12))

        customtkinter.CTkLabel(config_card, text="Model Download / Selection Location", font=("Segoe UI", 12, "bold"), text_color=TEXT).pack(anchor="w", padx=20, pady=(4, 4))
        path_row = customtkinter.CTkFrame(config_card, fg_color="transparent")
        path_row.pack(fill="x", padx=20, pady=(0, 16))

        self.path_entry = customtkinter.CTkEntry(path_row, height=38, corner_radius=9, fg_color=PANEL_ALT, border_color=BORDER, text_color=TEXT)
        self.path_entry.insert(0, self.selected_path)
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.browse_btn = customtkinter.CTkButton(
            path_row, text="Browse...", width=90, height=38, corner_radius=9,
            fg_color="#E2E8F0", hover_color="#CBD5E1", text_color=TEXT, font=("Segoe UI", 11, "bold"),
            command=self._browse_path
        )
        self.browse_btn.pack(side="right")

        logs_card = customtkinter.CTkFrame(self.scroll_frame, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        logs_card.pack(fill="x", padx=10, pady=10)

        status_bar_row = customtkinter.CTkFrame(logs_card, fg_color="transparent")
        status_bar_row.pack(fill="x", padx=20, pady=(12, 4))

        self.hw_status_lbl = customtkinter.CTkLabel(
            status_bar_row, text="Detecting Hardware Specifications...", font=("Segoe UI", 11, "bold"), text_color=MUTED
        )
        self.hw_status_lbl.pack(side="left")

        self.progress_lbl = customtkinter.CTkLabel(
            status_bar_row, text="0%", font=("Segoe UI", 11, "bold"), text_color=ACCENT
        )
        self.progress_lbl.pack(side="right")

        self.progress_bar = customtkinter.CTkProgressBar(logs_card, height=8, corner_radius=8, progress_color=ACCENT, fg_color="#E2E8F0")
        self.progress_bar.pack(fill="x", padx=20, pady=(4, 8))
        self.progress_bar.set(0.0)

        # Dynamic Download Bar Container (hidden until downloading)
        self.dl_container = customtkinter.CTkFrame(logs_card, fg_color="transparent")
        
        dl_text_row = customtkinter.CTkFrame(self.dl_container, fg_color="transparent")
        dl_text_row.pack(fill="x", pady=(2, 2))

        self.dl_status_lbl = customtkinter.CTkLabel(
            dl_text_row, text="⬇ Model Download Progress:", font=("Segoe UI", 10, "bold"), text_color=ACCENT
        )
        self.dl_status_lbl.pack(side="left")

        self.dl_percent_lbl = customtkinter.CTkLabel(
            dl_text_row, text="0%", font=("Segoe UI", 10, "bold"), text_color=ACCENT
        )
        self.dl_percent_lbl.pack(side="right")

        self.dl_progress_bar = customtkinter.CTkProgressBar(self.dl_container, height=6, corner_radius=6, progress_color="#10B981", fg_color="#E2E8F0")
        self.dl_progress_bar.pack(fill="x", pady=(0, 6))
        self.dl_progress_bar.set(0.0)

        self.log_box = customtkinter.CTkTextbox(logs_card, height=160, font=("Consolas", 10), fg_color="#F8FAFC", text_color="#334155", border_width=1, border_color=BORDER, corner_radius=9)
        self.log_box.pack(fill="x", padx=20, pady=(0, 14))

        action_card = customtkinter.CTkFrame(self.scroll_frame, fg_color="transparent")
        action_card.pack(fill="x", padx=10, pady=(4, 16))

        self.start_btn = customtkinter.CTkButton(
            action_card, text="Download / Run", height=46, corner_radius=11,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#FFFFFF", font=("Segoe UI", 12, "bold"),
            command=self._start_download_and_load
        )
        self.start_btn.pack(fill="x")

    def log(self, message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _set_progress(self, val):
        val = min(1.0, max(0.0, val))
        pct = int(val * 100)
        self.progress_bar.set(val)
        self.progress_lbl.configure(text=f"{pct}%")

    def _show_download_bar(self):
        self.dl_container.pack(fill="x", padx=20, pady=(0, 6), before=self.log_box)

    def _hide_download_bar(self):
        self.dl_container.pack_forget()

    def _format_size(self, size_bytes):
        if size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        return f"{size_bytes / (1024 ** 3):.2f} GB"

    def _set_download_progress(self, percentage, current_bytes=0, target_bytes=0):
        percentage = min(1.0, max(0.0, percentage))
        pct = int(percentage * 100)
        self.dl_progress_bar.set(percentage)

        if target_bytes > 0:
            cur_str = self._format_size(current_bytes)
            tot_str = self._format_size(target_bytes)
            label_text = f"{pct}% ({cur_str} / {tot_str})"
        else:
            label_text = f"{pct}%"

        self.dl_percent_lbl.configure(text=label_text)

    def _get_directory_size(self, path):
        total_size = 0
        if not os.path.exists(path):
            return 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def _monitor_download_progress(self, target_dir):
        if not self.is_downloading:
            return

        current_bytes = self._get_directory_size(target_dir)
        if self.download_target_size_bytes > 0:
            percentage = min(1.0, current_bytes / self.download_target_size_bytes)
            self._set_download_progress(percentage, current_bytes, self.download_target_size_bytes)
            scaled_val = 0.20 + (percentage * 0.55)
            self._set_progress(scaled_val)

        self.after(1000, lambda: self._monitor_download_progress(target_dir))

    def _fetch_model_size_dynamic(self, choice):
        def task():
            from huggingface_hub import HfApi
            repo_id = POPULAR_MODELS[choice]["repo"]
            try:
                self.log(f"Fetching metadata for '{repo_id}' from Hugging Face API...")
                api = HfApi()
                model_info = api.model_info(repo_id, files_metadata=True)
                
                download_extensions = (".safetensors", ".bin", ".json", ".model")
                actual_files = [
                    sibling.size for sibling in model_info.siblings 
                    if sibling.size is not None and sibling.rfilename.endswith(download_extensions)
                ]
                
                has_safetensors = any(s.rfilename.endswith(".safetensors") for s in model_info.siblings)
                if has_safetensors:
                    actual_files = [
                        sibling.size for sibling in model_info.siblings 
                        if sibling.size is not None and (sibling.rfilename.endswith(".safetensors") or sibling.rfilename.endswith(".json") or sibling.rfilename.endswith(".model"))
                    ]

                total_bytes = sum(actual_files)
                self.download_target_size_bytes = total_bytes
                size_gb = total_bytes / (1024 ** 3)
                req_space_gb = size_gb * 1.2
                
                text_out = f"Actual Download Size: {size_gb:.2f} GB  |  Estimated Disk Space Required: {req_space_gb:.1f} GB"
                self.after(0, lambda: self.model_info_lbl.configure(text=text_out, text_color=ACCENT))
                self.log(f"Successfully calculated download size for '{repo_id}': {size_gb:.2f} GB.")
            except Exception as err:
                self.after(0, lambda: self.model_info_lbl.configure(
                    text="Unable to query HF API dynamically. Estimated download: ~13.5 GB.", text_color=MUTED
                ))
                self.log(f"Metadata fetch fallback: {str(err)}")

        Thread(target=task, daemon=True).start()

    def _on_model_select(self, choice):
        if not POPULAR_MODELS[choice]["supported"]:
            messagebox.showwarning(
                "Model Unavailable",
                f"The selected model {choice} is currently not available."
            )
            self.model_dropdown.set(self.current_selected_model)
            return

        self.current_selected_model = choice
        self.model_info_lbl.configure(text="Calculating size from Hugging Face...", text_color=MUTED)
        self._fetch_model_size_dynamic(choice)

    def _browse_path(self):
        chosen = filedialog.askdirectory(initialdir=self.selected_path)
        if chosen:
            self.selected_path = chosen
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, chosen)

    def _check_hardware_requirements(self):
        global runtime_device
        import torch

        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        has_gpu = torch.cuda.is_available()
        vram_gb = 0.0

        if has_gpu:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        self.log("--- SYSTEM HARDWARE CHECK ---")
        self.log(f"System RAM: {ram_gb:.1f} GB")
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            self.log(f"GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            self.log("No CUDA GPU detected.")

        if ram_gb < 12.0 or not has_gpu or vram_gb < 6.0:
            reasons = []
            if ram_gb < 12.0:
                reasons.append(f"• System RAM ({ram_gb:.1f} GB) is less than 12 GB")
            if not has_gpu:
                reasons.append("• No dedicated CUDA GPU detected")
            elif vram_gb < 6.0:
                reasons.append(f"• GPU VRAM ({vram_gb:.1f} GB) is less than 6 GB")

            err_msg = "Incompatible Device Hardware:\n\n" + "\n".join(reasons)
            
            self.hw_status_lbl.configure(
                text="❌ Incompatible Device: Minimum 12GB RAM & 6GB VRAM GPU Required", text_color=RED
            )
            self.start_btn.configure(state="disabled", fg_color="#CBD5E1")
            self.log("Device hardware incompatible. CPU mode is unsupported.")
            messagebox.showerror("Incompatible Device", err_msg)
        else:
            runtime_device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            self.hw_status_lbl.configure(
                text=f"✓ Supported Device Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)", text_color=GREEN
            )
            self.log("System hardware meets minimum execution requirements.")

    def _start_download_and_load(self):
        chosen_model = self.model_dropdown.get()
        if not POPULAR_MODELS[chosen_model]["supported"]:
            messagebox.showwarning("Model Not Supported", "Only supported models are available.")
            return

        self.start_btn.configure(state="disabled", text="Loading Model...")
        self.model_dropdown.configure(state="disabled")

        self.path_entry.configure(state="disabled")
        self.browse_btn.configure(state="disabled", fg_color="#CBD5E1")

        target_dir = self.path_entry.get().strip()
        os.environ["HF_HOME"] = target_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(target_dir, "datasets")

        Thread(target=self._download_task, daemon=True).start()

    def _is_model_present_locally(self, target_dir):
        if not os.path.exists(target_dir):
            return False
        for root, _, files in os.walk(target_dir):
            for file in files:
                if file.endswith((".safetensors", ".bin")):
                    return True
        return False

    def _download_task(self):
        global tokenizer, model, stop_token_ids

        # Lazy imports inside background thread for fast instant GUI launch
        self.log("Initializing PyTorch and Hugging Face dependencies...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import snapshot_download

        base_model_name = "mistralai/Mistral-7B-v0.1"
        adapter_path = os.path.join(self.get_base_dir(), "my-mistral-chatbot")
        target_dir = os.environ["HF_HOME"]

        try:
            model_exists = self._is_model_present_locally(target_dir)

            if model_exists:
                self.log("Local model files detected on disk. Skipping download step...")
                self.after(0, lambda: self._set_progress(0.50))
            else:
                self.log("Model weights not found locally. Starting download pipeline...")
                self.is_downloading = True
                self.after(0, self._show_download_bar)
                self.after(0, lambda: self._set_download_progress(0.0, 0, self.download_target_size_bytes))
                self._monitor_download_progress(target_dir)
                self.after(0, lambda: self._set_progress(0.10))

            # 1. Tokenizer Download / Local Load
            self.log("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, 
                cache_dir=target_dir,
                local_files_only=model_exists
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            if not model_exists:
                self.after(0, lambda: self._set_progress(0.20))
                self.log("Downloading model weight files to target directory...")

            # 2. Get local snapshot path
            local_model_path = snapshot_download(
                repo_id=base_model_name,
                cache_dir=target_dir,
                local_files_only=model_exists,
                allow_patterns=[
                    "*.safetensors",
                    "*.json",
                    "*.model",
                    "tokenizer*"
                ]
            )

            if not model_exists:
                try:
                    if sys.__stdout__ is not None:
                        sys.__stdout__.write("\rModel download complete: 100%\n")
                        sys.__stdout__.flush()
                except Exception:
                    pass
                self.is_downloading = False
                self.after(0, lambda: self._set_download_progress(1.0, self.download_target_size_bytes, self.download_target_size_bytes))
                self.log("✓ Model weights downloaded successfully to disk!")
                time.sleep(0.5)
                self.after(0, self._hide_download_bar)
            else:
                self.log("✓ Loaded local model directory successfully!")

            self.after(0, lambda: self._set_progress(0.75))

            # 3. Quantize and load into GPU memory
            self.log("Loading model into GPU VRAM (4-bit NF4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )

            self.after(0, lambda: self._set_progress(0.90))
            self.log("Attaching custom trained LoRA adapters...")
            
            model = PeftModel.from_pretrained(base_model, adapter_path)

            im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            stop_token_ids = [tokenizer.eos_token_id]
            if im_end_tokens:
                stop_token_ids.append(im_end_tokens[0])

            self.after(0, lambda: self._set_progress(1.0))
            self.log("✓ Model weights and adapters loaded successfully!")
            time.sleep(1)

            self.after(0, self._launch_main_gui)

        except Exception as e:
            self.is_downloading = False
            self.after(0, self._hide_download_bar)
            self.log(f"❌ Error during loading: {str(e)}")
            self.after(0, lambda: self._reset_ui_on_failure())

    def _reset_ui_on_failure(self):
        self.start_btn.configure(state="normal", text="Retry Download & Start")
        self.model_dropdown.configure(state="normal")
        self.path_entry.configure(state="normal")
        self.browse_btn.configure(state="normal", fg_color="#E2E8F0")

    def _launch_main_gui(self):
        self.is_downloading = False
        if sys.__stdout__ is not None:
            sys.stdout = sys.__stdout__
        if sys.__stderr__ is not None:
            sys.stderr = sys.__stderr__
        self.destroy()
        app = AmpleFlowApp()
        app.mainloop()


class LicenseModal(customtkinter.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Software Licenses & Legal Compliance")
        self.geometry("620x540")
        self.configure(fg_color=BG)
        self.resizable(False, False)

        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (screen_w - 620) // 2, (screen_h - 540) // 2
        self.geometry(f"620x540+{x}+{y}")

        container = customtkinter.CTkFrame(self, fg_color=PANEL, corner_radius=16, border_width=1, border_color=BORDER)
        container.pack(fill="both", expand=True, padx=16, pady=16)

        customtkinter.CTkLabel(
            container, text="Model & Dataset Licenses", font=("Segoe UI", 16, "bold"), text_color=TEXT
        ).pack(anchor="w", padx=20, pady=(18, 4))

        textbox = customtkinter.CTkTextbox(
            container, font=("Segoe UI", 10.5), fg_color="#F8FAFC", text_color="#334155",
            border_width=1, border_color="#CBD5E1", corner_radius=10, wrap="word"
        )
        textbox.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        license_text = (
            "1. BASE MODEL LICENSE (Mistral-7B-v0.1)\n"
            "   • Released under the Apache 2.0 License.\n\n"
            "2. FINE-TUNING DATASET (OpenHermes-2.5)\n"
            "   • OpenHermes-2.5 comprises permissive datasets.\n\n"
            "3. ADAPTER WEIGHTS (LoRA Derivatives)\n"
            "   • Full legal rights retained by derivative creator."
        )
        textbox.insert("1.0", license_text)
        textbox.configure(state="disabled")

        close_btn = customtkinter.CTkButton(
            container, text="Close", width=110, height=36, corner_radius=9,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="#FFFFFF",
            command=self.destroy
        )
        close_btn.pack(anchor="e", padx=20, pady=(0, 18))
        self.attributes("-topmost", True)


class AmpleFlowApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("AmpleFlow (Your Custom Pocket LLM)")
        self.geometry("1280x820")
        self.minsize(1050, 700)
        self.configure(fg_color=BG)

        self._center_window(1280, 820)
        self.history = []
        self.current_page = "chat"
        self._page_transition_running = False

        self.build_ui()

    def _center_window(self, width, height):
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        x, y = (screen_w - width) // 2, (screen_h - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def get_base_dir(self):
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    def build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = customtkinter.CTkFrame(
            self, width=230, corner_radius=0, fg_color=SIDEBAR
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self._build_sidebar()

        self.workspace = customtkinter.CTkFrame(
            self, fg_color=BG, corner_radius=0
        )
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=1)

        self._build_topbar()

        self.page_host = customtkinter.CTkFrame(
            self.workspace, fg_color=BG, corner_radius=0
        )
        self.page_host.grid(row=1, column=0, sticky="nsew", padx=24, pady=(8, 22))
        self.page_host.grid_columnconfigure(0, weight=1)
        self.page_host.grid_rowconfigure(0, weight=1)

        self.pages = {}
        self.pages["chat"] = self.build_chat_page()
        self.pages["settings"] = self.build_settings_page()

        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            page.lower()

        self.pages["chat"].lift()
        self._set_active_nav("chat")

    def _build_sidebar(self):
        brand = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=20, pady=(25, 30))

        # --- LOGO INTEGRATION (MAIN SIDEBAR) ---
        logo_path = os.path.abspath(os.path.join(self.get_base_dir(), ".Icons", "1.png"))
        if os.path.exists(logo_path):
            sidebar_img = Image.open(logo_path)
            self.sidebar_logo = customtkinter.CTkImage(light_image=sidebar_img, dark_image=sidebar_img, size=(42, 42))
            mark = customtkinter.CTkLabel(
                brand, image=self.sidebar_logo, text="", width=42, height=42
            )
        else:
            mark = customtkinter.CTkLabel(
                brand, text="A", width=42, height=42, corner_radius=13,
                fg_color=ACCENT, text_color="#FFFFFF",
                font=("Segoe UI", 22, "bold")
            )
        mark.pack(side="left")

        brand_text = customtkinter.CTkFrame(brand, fg_color="transparent")
        brand_text.pack(side="left", padx=11)
        customtkinter.CTkLabel(
            brand_text, text="AmpleFlow", font=("Segoe UI", 16, "bold"),
            text_color=TEXT
        ).pack(anchor="w")
        customtkinter.CTkLabel(
            brand_text, text="POCKET LLM", font=("Segoe UI", 9, "bold"),
            text_color=ACCENT
        ).pack(anchor="w", pady=(1, 0))

        self.nav_buttons = {}
        nav_items = [
            ("chat", "◉", "Chat", "Interactive Assistant"),
            ("settings", "⚙", "Configuration", "Persona & behavior"),
        ]
        for key, icon, title, subtitle in nav_items:
            btn = customtkinter.CTkButton(
                self.sidebar, text=f"{icon}   {title}",
                anchor="w", height=48, corner_radius=12,
                fg_color="transparent", hover_color="#E2E8F0",
                text_color="#334155", font=("Segoe UI", 13, "bold"),
                command=lambda k=key: self.switch_page(k)
            )
            btn.pack(fill="x", padx=13, pady=4)
            self.nav_buttons[key] = btn

        spacer = customtkinter.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)

        license_btn = customtkinter.CTkButton(
            self.sidebar, text="⚖  Licenses & Legal", height=36, corner_radius=9,
            fg_color="#E2E8F0", hover_color="#CBD5E1", text_color="#334155",
            font=("Segoe UI", 10, "bold"), command=self.open_licenses
        )
        license_btn.pack(fill="x", padx=16, pady=(0, 12))

        customtkinter.CTkLabel(
            self.sidebar, text=f"AmpleFlow  •  {runtime_device.upper()} Mode",
            font=("Segoe UI", 9), text_color="#64748B"
        ).pack(anchor="w", padx=22, pady=(0, 20))

    def open_licenses(self):
        LicenseModal(self)

    def _build_topbar(self):
        topbar = customtkinter.CTkFrame(self.workspace, fg_color="transparent")
        topbar.grid(row=0, column=0, sticky="ew", padx=26, pady=(22, 8))
        topbar.grid_columnconfigure(0, weight=1)

        heading = customtkinter.CTkFrame(topbar, fg_color="transparent")
        heading.grid(row=0, column=0, sticky="w")
        self.page_title = customtkinter.CTkLabel(
            heading, text="Chat", font=("Segoe UI", 25, "bold"),
            text_color=TEXT
        )
        self.page_title.pack(anchor="w")
        self.page_subtitle = customtkinter.CTkLabel(
            heading, text="Talk to your locally fine-tuned assistant.",
            font=("Segoe UI", 11), text_color=MUTED
        )
        self.page_subtitle.pack(anchor="w", pady=(3, 0))

        self.app_title = self.page_title
        self.app_desc = self.page_subtitle

    def _set_active_nav(self, active_key):
        for key, button in self.nav_buttons.items():
            if key == active_key:
                button.configure(
                    fg_color="#DBEAFE", text_color="#1E40AF",
                    hover_color="#BFDBFE"
                )
            else:
                button.configure(
                    fg_color="transparent", text_color="#334155",
                    hover_color="#E2E8F0"
                )

    def switch_page(self, page_key):
        if page_key == self.current_page or self._page_transition_running:
            return

        self._page_transition_running = True
        old_page = self.pages[self.current_page]
        new_page = self.pages[page_key]

        titles = {
            "chat": ("Chat", "Talk to your locally fine-tuned assistant."),
            "settings": ("Configuration", "Personalize assistant identity and response behavior."),
        }
        self.page_title.configure(text=titles[page_key][0])
        self.page_subtitle.configure(text=titles[page_key][1])
        self._set_active_nav(page_key)

        width = max(1, self.page_host.winfo_width())
        new_page.place(x=width, y=0, relwidth=1, relheight=1)
        new_page.lift()

        steps = 14
        def animate(step=0):
            if step >= steps:
                old_page.lower()
                new_page.place(x=0, y=0, relwidth=1, relheight=1)
                self.current_page = page_key
                self._page_transition_running = False
                return
            progress = step / steps
            eased = 1 - (1 - progress) ** 3
            offset = int(width * (1 - eased))
            new_page.place(x=offset, y=0, relwidth=1, relheight=1)
            old_page.place(x=int(-width * eased * 0.18), y=0, relwidth=1, relheight=1)
            self.after(12, lambda: animate(step + 1))
        animate()

    def build_chat_page(self):
        page = customtkinter.CTkFrame(self.page_host, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)

        disclaimer = customtkinter.CTkFrame(page, fg_color="#FEF3C7", corner_radius=10, border_width=1, border_color="#FCD34D")
        disclaimer.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        customtkinter.CTkLabel(
            disclaimer,
            text="⚠️ Disclaimer: AI responses may be inaccurate or hallucinated. Verify crucial information independently.",
            font=("Segoe UI", 10, "bold"), text_color="#92400E"
        ).pack(anchor="w", padx=14, pady=8)

        toolbar = customtkinter.CTkFrame(page, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        toolbar.grid_columnconfigure(0, weight=1)


        self.clear_btn = customtkinter.CTkButton(
            toolbar, text="⌫  Clear conversation", width=155, height=34,
            corner_radius=9, fg_color="#E2E8F0", hover_color="#CBD5E1",
            text_color="#334155", font=("Segoe UI", 10, "bold"),
            command=self.clear_chat
        )
        self.clear_btn.grid(row=0, column=1, sticky="e")

        chat_card = customtkinter.CTkFrame(
            page, fg_color=PANEL, corner_radius=18,
            border_width=1, border_color=BORDER
        )
        chat_card.grid(row=2, column=0, sticky="nsew")
        chat_card.grid_columnconfigure(0, weight=1)
        chat_card.grid_rowconfigure(0, weight=1)

        self.chat_display = customtkinter.CTkTextbox(
            chat_card, font=("Segoe UI", 12), wrap="word",
            fg_color=PANEL, text_color="#0F172A",
            border_width=0, corner_radius=14,
            scrollbar_button_color="#CBD5E1",
            scrollbar_button_hover_color=ACCENT
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=18, pady=(15, 10))

        self.chat_display.tag_config("user", foreground=ACCENT)
        self.chat_display.tag_config("bot", foreground=GREEN)
        self.chat_display.tag_config("welcome", foreground="#64748B")

        self.chat_display.configure(state="disabled")

        input_panel = customtkinter.CTkFrame(
            page, fg_color=PANEL_ALT, corner_radius=16,
            border_width=1, border_color=BORDER
        )
        input_panel.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        input_panel.grid_columnconfigure(0, weight=1)

        self.msg_entry = customtkinter.CTkEntry(
            input_panel, placeholder_text="Message AmpleFlow...",
            font=("Segoe UI", 12), height=48, corner_radius=11,
            fg_color="#FFFFFF", border_color="#CBD5E1",
            text_color=TEXT, placeholder_text_color="#94A3B8"
        )
        self.msg_entry.grid(row=0, column=0, sticky="ew", padx=(12, 8), pady=12)
        self.msg_entry.bind("<Return>", lambda event: self.send_message())

        self.send_btn = customtkinter.CTkButton(
            input_panel, text="Send  ➤", width=112, height=48,
            corner_radius=11, font=("Segoe UI", 12, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="#FFFFFF", command=self.send_message
        )
        self.send_btn.grid(row=0, column=1, padx=(0, 12), pady=12)

        self.chat_display.configure(state="normal")
        self.chat_display.insert(
            "end",
            "Your offline assistant is ready. Enter a message below to begin.\n\n"
        )
        self.chat_display.configure(state="disabled")
        return page

    def build_settings_page(self):
        page = customtkinter.CTkFrame(self.page_host, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)

        scroll = customtkinter.CTkScrollableFrame(
            page, fg_color="transparent", corner_radius=0,
            scrollbar_button_color="#CBD5E1",
            scrollbar_button_hover_color=ACCENT
        )
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)
        scroll.grid_columnconfigure(1, weight=1)

        identity = customtkinter.CTkFrame(
            scroll, fg_color=PANEL, corner_radius=16,
            border_width=1, border_color=BORDER
        )
        identity.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        identity.grid_columnconfigure(0, weight=1)
        identity.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(
            identity, text="Assistant identity", font=("Segoe UI", 15, "bold"),
            text_color=TEXT
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 2))

        self._field_label(identity, "Assistant name", 2, 0)
        self.name_entry = customtkinter.CTkEntry(
            identity, height=42, corner_radius=10,
            fg_color="#F8FAFC", border_color="#CBD5E1",
            font=("Segoe UI", 12)
        )
        self.name_entry.insert(0, "AmpleFlow Assistant")
        self.name_entry.grid(row=3, column=0, sticky="ew", padx=(18, 8), pady=(5, 16))

        self._field_label(identity, "Description / subtitle", 2, 1)
        self.desc_entry = customtkinter.CTkEntry(
            identity, height=42, corner_radius=10,
            fg_color="#F8FAFC", border_color="#CBD5E1",
            font=("Segoe UI", 12)
        )
        self.desc_entry.insert(0, "Powered by fine-tuned local LoRA model")
        self.desc_entry.grid(row=3, column=1, sticky="ew", padx=(8, 18), pady=(5, 16))

        prompt_card = self._settings_card(scroll, "System prompt", "Set the assistant's role, tone, and response guidelines.")
        prompt_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.sys_prompt_entry = customtkinter.CTkTextbox(
            prompt_card, height=130, corner_radius=10,
            fg_color="#F8FAFC", border_width=1, border_color="#CBD5E1",
            font=("Segoe UI", 12), wrap="word"
        )
        self.sys_prompt_entry.insert(
            "1.0",
            "You are a highly intelligent, logical AI assistant named AmpleFlow."
        )
        self.sys_prompt_entry.pack(fill="x", padx=18, pady=(4, 18))

        generation = self._settings_card(scroll, "Response behavior", "Customize how the AI formulates its answers.")
        generation.grid(row=2, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
        generation.grid_columnconfigure(0, weight=1)

        customtkinter.CTkLabel(
            generation, text="Response Creativity", font=("Segoe UI", 12, "bold"),
            text_color=TEXT
        ).pack(anchor="w", padx=18, pady=(8, 2))
        
        temp_row = customtkinter.CTkFrame(generation, fg_color="transparent")
        temp_row.pack(fill="x", padx=18, pady=(8, 16))
        self.temp_slider = customtkinter.CTkSlider(
            temp_row, from_=0.1, to=1.5, number_of_steps=28,
            progress_color=ACCENT, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER
        )
        self.temp_slider.set(0.3)
        self.temp_slider.pack(side="left", fill="x", expand=True)
        self.temp_value = customtkinter.CTkLabel(
            temp_row, text="Focused (0.30)", width=110, font=("Segoe UI", 10, "bold"),
            text_color=ACCENT
        )
        self.temp_value.pack(side="right", padx=(10, 0))

        customtkinter.CTkLabel(
            generation, text="Vocabulary Diversity", font=("Segoe UI", 12, "bold"),
            text_color=TEXT
        ).pack(anchor="w", padx=18, pady=(0, 2))
        
        top_row = customtkinter.CTkFrame(generation, fg_color="transparent")
        top_row.pack(fill="x", padx=18, pady=(8, 16))
        self.topp_slider = customtkinter.CTkSlider(
            top_row, from_=0.1, to=1.0, number_of_steps=18,
            progress_color=ACCENT, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER
        )
        self.topp_slider.set(0.9)
        self.topp_slider.pack(side="left", fill="x", expand=True)
        self.topp_value = customtkinter.CTkLabel(
            top_row, text="Varied (0.90)", width=110, font=("Segoe UI", 10, "bold"),
            text_color=ACCENT
        )
        self.topp_value.pack(side="right", padx=(10, 0))

        save_bar = customtkinter.CTkFrame(scroll, fg_color="transparent")
        save_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.save_settings_btn = customtkinter.CTkButton(
            save_bar, text="Apply settings", height=44, width=190,
            corner_radius=11, font=("Segoe UI", 12, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="#FFFFFF", command=self.apply_settings
        )
        self.save_settings_btn.pack(side="right")
        return page

    def _settings_card(self, parent, title, subtitle):
        card = customtkinter.CTkFrame(
            parent, fg_color=PANEL, corner_radius=16,
            border_width=1, border_color=BORDER
        )
        customtkinter.CTkLabel(
            card, text=title, font=("Segoe UI", 15, "bold"),
            text_color=TEXT
        ).pack(anchor="w", padx=18, pady=(16, 3))
        customtkinter.CTkLabel(
            card, text=subtitle, font=("Segoe UI", 10),
            text_color=MUTED
        ).pack(anchor="w", padx=18, pady=(0, 12))
        return card

    def _field_label(self, parent, text, row, column):
        customtkinter.CTkLabel(
            parent, text=text, font=("Segoe UI", 11, "bold"),
            text_color="#475569"
        ).grid(row=row, column=column, sticky="w",
               padx=(18, 8) if column == 0 else (8, 18), pady=(0, 0))

    def apply_settings(self):
        name = self.name_entry.get().strip()
        desc = self.desc_entry.get().strip()
        if name:
            self.app_title.configure(text=name)
        if desc:
            self.app_desc.configure(text=desc)

    def clear_chat(self):
        self.history.clear()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

    def send_message(self):
        msg = self.msg_entry.get().strip()
        if not msg:
            return

        self.msg_entry.delete(0, "end")
        self.send_btn.configure(state="disabled", text="Thinking…")

        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "You\n", "user")
        self.chat_display.insert("end", f"{msg}\n\n")
        bot_title = self.app_title.cget("text")
        self.chat_display.insert("end", f"{bot_title}\n", "bot")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

        Thread(target=self.generate_response_thread, args=(msg,), daemon=True).start()

    def generate_response_thread(self, current_msg):
        from transformers import TextIteratorStreamer
        import torch

        sys_prompt = self.sys_prompt_entry.get("1.0", "end").strip()
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"

        for user_h, bot_h in self.history:
            prompt += f"<|im_start|>user\n{user_h}<|im_end|>\n<|im_start|>assistant\n{bot_h}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{current_msg}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=float(self.temp_slider.get()),
            top_p=float(self.topp_slider.get()),
            repetition_penalty=1.1,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )

        t = Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
        t.start()

        full_bot_response = ""
        for new_text in streamer:
            clean_text = new_text.replace("<|im_start|>", "").replace("<|im_end|>", "")
            if clean_text:
                full_bot_response += clean_text
                self.after(0, self.append_bot_token, clean_text)

        self.history.append((current_msg, full_bot_response))
        self.after(0, self.finish_bot_stream)

    def append_bot_token(self, token):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", token)
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def finish_bot_stream(self):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        self.send_btn.configure(state="normal", text="Send  ➤")


if __name__ == "__main__":
    init_window = ModelSelectionWindow()
    init_window.mainloop()