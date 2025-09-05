"""
下載並驗證 CKIP 斷詞任務頭（WS）
- 目標 Repo : ckiplab/bert-base-chinese-ws
- 下載到    : models/CKIP/ws
- HF 快取   : models/CKIP（集中管理，便於離線/部署）

使用方式：
  python models/download_ckip-ws.py              # 偵測→（必要時）下載→驗證
  python models/download_ckip-ws.py --no-verify  # 僅偵測/下載，不做驗證
  python models/download_ckip-ws.py --offline    # 離線模式（需已快取）
  python models/download_ckip-ws.py --device 0   # 驗證時鎖 GPU0

可選參數：
  --repo-id, --target-dir, --offline, --device, --no-verify, --force-download
"""

from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import sys
from typing import Iterable

# --------------------------
# 環境與常數
# --------------------------
CACHE_ROOT = Path("models") / "CKIP"        # HF_HOME / TRANSFORMERS_CACHE
WS_DIR = Path("models") / "CKIP" / "ws"  # 下載目標資料夾
DEFAULT_REPO = "ckiplab/bert-base-chinese-ws"


def set_hf_env(cache_root: Path, offline: bool) -> None:
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"


def ensure_deps() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        print(
            "[錯誤] 未安裝 huggingface_hub，請先執行：pip install -U huggingface_hub", file=sys.stderr)
        sys.exit(1)


def has_ws_local(target_dir: Path) -> bool:
    """粗略檢查 WS 目錄是否『看起來完整』：至少要有 config/tokenizer 與權重檔。"""
    if not target_dir.exists():
        return False
    files = [p.name for p in target_dir.glob("*") if p.is_file()]
    if not files:
        return False
    needed_any = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"
    ]
    has_meta = any(n in files for n in needed_any)
    has_weight = any(
        n.startswith(("pytorch_model", "model")) and n.endswith(
            (".bin", ".safetensors"))
        for n in files
    )
    return has_meta and has_weight


def snapshot_fetch(repo_id: str, target_dir: Path) -> Path:
    from huggingface_hub import snapshot_download
    target_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        revision="main",
    )
    return Path(path)

# --------------------------
# GPU 自動偵測（WSL 友善）
# --------------------------


def auto_detect_device() -> int:
    """回傳 device：-1=CPU；0=GPU0（若可用）"""
    env = os.environ.get("CKIP_DEVICE")
    if env is not None:
        try:
            return int(env.strip())
        except ValueError:
            pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        return 0
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_vis and cuda_vis.strip() not in {"", "-1"}:
        return 0
    return -1

# --------------------------
# 驗證
# --------------------------


def verify_ws(device: int, model_name: str) -> None:
    try:
        from ckip_transformers.nlp import CkipWordSegmenter  # type: ignore
    except Exception:
        print("[警告] 未安裝 ckip-transformers，略過驗證。若需驗證請安裝：pip install ckip-transformers")
        return
    print(
        f"[驗證] 初始化 CkipWordSegmenter(model_name='{model_name}', device={device}) ...")
    ws = CkipWordSegmenter(model_name=model_name, device=device)
    toks = ws(["中央社今晨發布測試訊息，行政院表示方案將滾動檢討。"])
    print("[驗證] 斷詞結果：", toks[0])

# --------------------------
# 主程式
# --------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="下載並驗證 CKIP 斷詞任務頭（WS）")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--target-dir", default=str(WS_DIR))
    parser.add_argument("--offline", action="store_true", help="離線模式（需已快取）")
    parser.add_argument("--device", type=int, default=None,
                        help="-1=CPU, 0=GPU0；預設自動偵測")
    parser.add_argument("--no-verify", action="store_true", help="不進行驗證")
    parser.add_argument("--force-download",
                        action="store_true", help="無論是否存在皆重新下載")
    args = parser.parse_args()

    cache_root = CACHE_ROOT
    target_dir = Path(args.target_dir)

    set_hf_env(cache_root, offline=args.offline)
    ensure_deps()

    # 1) 偵測是否已有本地 WS
    already = has_ws_local(target_dir)
    if already and not args.force_download:
        print(f"[資訊] 偵測到本地 WS 已存在：{target_dir}，略過下載。")
    else:
        if args.offline and not already:
            print("[錯誤] 離線模式啟用但本地無 WS 快取。請移除 --offline 或先行放置快取。", file=sys.stderr)
            sys.exit(2)
        print(f"[資訊] 下載 {args.repo_id} 到 {target_dir} ...")
        try:
            local_path = snapshot_fetch(args.repo_id, target_dir)
        except Exception as e:
            print("[錯誤] 下載失敗：", e, file=sys.stderr)
            sys.exit(2)
        print(f"[完成] 斷詞任務頭已就緒：{local_path}")

    # 2) 驗證（預設啟用，可用 --no-verify 關閉）
    if not args.no_verify:
        dev = args.device if args.device is not None else auto_detect_device()
        # 直接用本地資料夾作為 model_name，確保走本地
        verify_ws(device=dev, model_name=str(target_dir))


if __name__ == "__main__":
    main()
