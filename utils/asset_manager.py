# RWKV-Scout/utils/asset_manager.py
import os
import json
import threading
from config import DATA_PIPELINE

_asset_lock = threading.Lock()

def get_asset_index_path():
    base_dir = DATA_PIPELINE.get("asset_directory", os.path.join(os.path.dirname(DATA_PIPELINE.get("output_directory", "./data/output")), "knowledge_assets"))
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "asset_index.json")

def load_asset_index():
    path = get_asset_index_path()
    idx = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try: idx = json.load(f)
            except: idx = {}
            
    # 🔴 同步校验：如果原溯源文件已消失（被用户删除），则直接抹除对应的物理资产和绑定记录
    changed = False
    dead_keys = []
    for src_path, asset in idx.items():
        if not os.path.exists(src_path):
            asset_path = asset.get("asset_path")
            if asset_path and os.path.exists(asset_path):
                try: os.remove(asset_path)
                except Exception: pass
            dead_keys.append(src_path)
            changed = True
            print(f"🧹 [资产清理] 溯源文件 {os.path.basename(src_path)} 已消失，同步注销其结构化资产。")
            
    for k in dead_keys:
        del idx[k]
        
    if changed:
        save_asset_index_unlocked(idx)
        
    return idx

def save_asset_index_unlocked(index):
    with open(get_asset_index_path(), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def bind_asset(source_path, asset_path, main_cat, sub_cat):
    """绑定原始溯源文件和最终资产文件，并记录文件的修改时间防止过期"""
    with _asset_lock:
        idx = load_asset_index()
        safe_source = os.path.abspath(source_path)
        try: mtime = os.path.getmtime(safe_source)
        except: mtime = 0
            
        idx[safe_source] = {
            "asset_path": asset_path,
            "main_cat": main_cat,
            "sub_cat": sub_cat,
            "mtime": mtime
        }
        save_asset_index_unlocked(idx)

def get_asset(source_path):
    """获取资产。如果溯源文件已被修改（mtime 发生变化），则返回 None 使资产失效"""
    with _asset_lock:
        idx = load_asset_index()
        safe_source = os.path.abspath(source_path)
        asset = idx.get(safe_source)
        
        if asset:
            try: current_mtime = os.path.getmtime(safe_source)
            except: current_mtime = 0
            
            # 🔴 如果原文件被修改了内容，资产即刻失效并被物理删除
            if current_mtime > asset.get("mtime", 0):
                asset_path = asset.get("asset_path")
                if asset_path and os.path.exists(asset_path):
                    try: os.remove(asset_path)
                    except Exception: pass
                del idx[safe_source]
                save_asset_index_unlocked(idx)
                print(f"🔄 [资产更新] 溯源文件 {os.path.basename(safe_source)} 发生变更，旧资产已作废。")
                return None
                
        return asset

def get_all_categories():
    """获取目前已有的全部分类大纲，供大模型判断"""
    with _asset_lock:
        idx = load_asset_index()
        tree = {}
        for asset in idx.values():
            mc = asset.get("main_cat", "综合领域")
            sc = asset.get("sub_cat", "默认分类")
            if mc not in tree: tree[mc] = []
            if sc not in tree[mc]: tree[mc].append(sc)
        return tree