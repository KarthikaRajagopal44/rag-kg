# patch_streamlit.py
import streamlit.watcher.local_sources_watcher

def patch_streamlit_file_watcher():
    # Monkey-patch to avoid PyTorch crash
    streamlit.watcher.local_sources_watcher.get_module_paths = lambda *_: []