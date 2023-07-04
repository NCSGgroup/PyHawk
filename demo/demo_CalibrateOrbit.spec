# -*- mode: python ; coding: utf-8 -*-


block_cipher = pyi_crypto.PyiBlockCipher(key='mypassword')


a = Analysis(
    ['demo_CalibrateOrbit.py'],
    pathex=['D:\\py-hawk-v2','D:\\py-hawk-v2\\src'],
    binaries=[],
    datas=[("D:\\py-hawk-v2\\setting", "setting"),("D:\\py-hawk-v2\\data\\eop", "data\\eop")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='demo_CalibrateOrbit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
