import os
import sys

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# NOTE: upstream set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 unconditionally,
# which disables MPS's OOM guard entirely. FoocusRX picks a saner default
# in modules.apple_silicon.configure() based on total unified memory, but
# preserves any explicit override the user has set.
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

# NOTE: Upstream Fooocus disables TLS certificate verification globally here
# via ssl._create_default_https_context = ssl._create_unverified_context. That
# turns every outbound HTTPS request in the process (PyTorch model downloads,
# HuggingFace Hub, rembg, etc.) into a MITM opportunity. FoocusRX removes it.
# If a user hits an SSL error behind a corporate CA, set SSL_CERT_FILE or
# REQUESTS_CA_BUNDLE to the CA bundle path instead.

# Apple Silicon chip detection + MPS tuning. No-op on non-Mac. Must run
# before anything imports torch so that PYTORCH_MPS_* env vars take effect.
try:
    from modules.apple_silicon import configure as _configure_apple_silicon
    _configure_apple_silicon()
except Exception as _e:
    # Never let detection failure block launch.
    print(f"[apple_silicon] detection skipped: {_e!r}")

import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def _default_torch_command():
    """Pick a torch install command that matches the platform.

    Upstream Fooocus hard-coded a CUDA 12.1 index URL and torch==2.1.0.
    On Apple Silicon that pulls the CPU wheel (no MPS-optimized build)
    and on any newer chip generation the pinned 2.1.0 is out of date.

    FoocusRX behavior:
      * Apple Silicon (arm64 Darwin) -> torch>=2.5 from PyPI (no index URL).
        2.5 is the first release with torch.mps.compile_shader, which the
        M5-class Flash Attention kernels (mtlflashattn, etc.) depend on.
      * Intel Mac -> torch>=2.5 from PyPI, CPU wheel.
      * Linux/Windows -> keep the CUDA 12.1 wheel as before (unchanged).
    Users can still override with TORCH_COMMAND / TORCH_INDEX_URL.
    """
    if os.environ.get('TORCH_COMMAND'):
        return os.environ['TORCH_COMMAND']

    if platform.system() == "Darwin":
        # PyPI ships MPS-enabled arm64 wheels; no --extra-index-url needed.
        return "pip install --upgrade 'torch>=2.5,<3' 'torchvision>=0.20,<1'"

    # Non-Mac: keep the historical CUDA 12.1 default so we don't break
    # existing NVIDIA installs.
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    return f"pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url {torch_index_url}"


def prepare_environment():
    torch_command = _default_torch_command()
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    # Auto-install mtlflashattn on Apple Silicon (M4+). No-ops on
    # non-Mac, on chips older than M4, if the shim is already
    # installed, if the user set FOOOCUS_AUTO_INSTALL_MTLFLASHATTN=0,
    # or if a marker file from a previous attempt is present. Never
    # raises — install failures print a message and continue.
    try:
        from modules.apple_silicon import auto_install_mtlflashattn as _auto_install_mfa
        status = _auto_install_mfa()
        if status not in ("not-apple-silicon", "gen-too-low", "already-installed"):
            print(f"[apple_silicon] mtlflashattn auto-install: {status}")
        # If we just installed it, run the activation path so the shim
        # is live in the current process instead of waiting for the
        # next launch to pick it up.
        if status == "installed":
            from modules.apple_silicon import _try_activate_mtlflashattn as _activate
            print(f"[apple_silicon] mtlflashattn activation: {_activate()}")
    except Exception as exc:
        print(f"[apple_silicon] mtlflashattn auto-install skipped: {exc!r}")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, config.paths_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, config.paths_checkpoints)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

from webui import *
