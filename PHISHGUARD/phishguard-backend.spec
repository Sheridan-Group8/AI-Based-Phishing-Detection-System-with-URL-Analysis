# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_dir = Path(SPECPATH)

datas = [
    (str(project_dir / "templates"), "templates"),
    (str(project_dir / "static"), "static"),
    (str(project_dir / "config"), "config"),
    (str(project_dir / "bigmodel" / "onnx-model"), "bigmodel/onnx-model"),
    (str(project_dir / "bigmodel" / "onnx-url-model"), "bigmodel/onnx-url-model"),
]

a = Analysis(
    ["app.py"],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        "tokenizers",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="phishguard-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # Electron supplies a private stdin pipe for live provider configuration.
    # Keep standard streams available; main.js uses windowsHide so this does
    # not display a console window in the packaged desktop app.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="phishguard-backend",
)
