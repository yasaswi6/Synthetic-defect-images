"""Extract a multi-volume RAR (e.g. RIAWELC_dataset.partXX.rar) using the system libarchive.

Usage: python unrar_multi.py <folder_with_parts> <output_folder>
(On Windows you can just open part01.rar with 7-Zip or WinRAR instead.)
"""
import ctypes
import os
import sys
from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_void_p

src, out = sys.argv[1], sys.argv[2]
lib = ctypes.CDLL("libarchive.so.13")
lib.archive_read_new.restype = c_void_p
lib.archive_read_support_format_all.argtypes = [c_void_p]
lib.archive_read_support_filter_all.argtypes = [c_void_p]
lib.archive_read_open_filenames.argtypes = [c_void_p, POINTER(c_char_p), c_size_t]
lib.archive_read_next_header.argtypes = [c_void_p, POINTER(c_void_p)]
lib.archive_entry_pathname.argtypes = [c_void_p]
lib.archive_entry_pathname.restype = c_char_p
lib.archive_entry_filetype.argtypes = [c_void_p]
lib.archive_entry_filetype.restype = c_int
lib.archive_read_data.argtypes = [c_void_p, c_void_p, c_size_t]
lib.archive_read_data.restype = ctypes.c_ssize_t
lib.archive_error_string.argtypes = [c_void_p]
lib.archive_error_string.restype = c_char_p

parts = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".rar"))
arr = (c_char_p * (len(parts) + 1))(*[p.encode() for p in parts], None)
a = lib.archive_read_new()
lib.archive_read_support_format_all(a)
lib.archive_read_support_filter_all(a)
r = lib.archive_read_open_filenames(a, arr, 10240)
print("open", r, lib.archive_error_string(a))
ent = c_void_p()
n = 0
buf = ctypes.create_string_buffer(1 << 20)
while True:
    r = lib.archive_read_next_header(a, byref(ent))
    if r == 1:  # ARCHIVE_EOF
        break
    if r < -20:
        print("header error", r, lib.archive_error_string(a))
        break
    p = lib.archive_entry_pathname(ent).decode("utf-8", "replace").replace("\\", "/")
    dest = os.path.join(out, p)
    if lib.archive_entry_filetype(ent) == 0o040000:
        os.makedirs(dest, exist_ok=True)
        continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as fh:
        while True:
            k = lib.archive_read_data(a, buf, len(buf))
            if k <= 0:
                if k < 0:
                    print("data error", p, lib.archive_error_string(a))
                break
            fh.write(buf.raw[:k])
    n += 1
print("files", n)
