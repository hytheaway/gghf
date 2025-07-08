from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {
    "packages": ["netCDF4.utils", "cftime", "darkdetect"],
    "excludes": [],
    "includes": [],
    "include_files": ["happyday.png"],
}

base = "gui"

executables = [Executable("main.py", base=base, target_name="GGHSF")]

setup(
    name="GGHSF",
    version="0.0.1",
    description="Garrett's Great HRTF & SOFA Functions",
    options={
        "build_exe": build_options,
        "bdist_mac": {"include_resources": [("happyday.png", "share/happyday.png")]},
    },
    executables=executables,
)
