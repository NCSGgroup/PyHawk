# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['demo_test_Coordinate.py'],
    pathex=['D:\py-hawk-v2','D:\py-hawk-v2\src'],
    binaries=[("D:\py-hawk-v2\lib\GravHar.dll","."),("D:\py-hawk-v2\lib\sofa.dll",".")],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},w
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
    name='demo_test_Coordinate',
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
    entitlements_file=None
)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='demo_test_Coordinate')