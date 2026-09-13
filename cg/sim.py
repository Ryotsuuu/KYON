import ctypes
import os
import platform
    
class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]

class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int)
    ]

os_name = platform.system()
cur_dir = os.path.dirname(os.path.abspath(__file__))

if os_name == 'Windows':
    lib_filename = "cg.dll"
elif os_name == "Darwin":
    lib_filename = "libcg.dylib"
elif platform.machine() in ('arm64', 'aarch64'):
    lib_filename = "libcg-arm64.so"
else:
    lib_filename = "libcg.so"

lib_path = os.path.join(cur_dir, lib_filename)
if not os.path.exists(lib_path):
    # Search common Kaggle and local paths
    search_paths = [
        os.path.join("/kaggle_simulations/agent/cg", lib_filename),
        os.path.join("/kaggle_simulations/agent", lib_filename),
        os.path.join(".", "cg", lib_filename),
        os.path.join(".", lib_filename),
    ]
    for sp in search_paths:
        if os.path.exists(sp):
            lib_path = sp
            break

lib = ctypes.cdll.LoadLibrary(lib_path)
try:
    lib.GameInitialize()
except Exception:
    pass

lib.BattleStart.restype = StartData
lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

lib.AgentStart.restype = ctypes.c_void_p

lib.BattleFinish.argtypes = [ctypes.c_void_p]

lib.GetBattleData.restype = SerialData
lib.GetBattleData.argtypes = [ctypes.c_void_p]

lib.Select.restype = ctypes.c_int
lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.VisualizeData.restype = ctypes.c_char_p
lib.VisualizeData.argtypes = [ctypes.c_void_p]

lib.SearchBegin.restype = ctypes.c_char_p
lib.SearchBegin.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int]

lib.SearchStep.restype = ctypes.c_char_p
lib.SearchStep.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.SearchEnd.argtypes = [ctypes.c_void_p]

lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]

lib.AllCard.restype = ctypes.c_char_p

lib.AllAttack.restype = ctypes.c_char_p

import threading

class BattleLocal(threading.local):
    def __init__(self):
        self.battle_ptr = None
        self.obs = None

Battle = BattleLocal()
