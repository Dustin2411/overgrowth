# Linux Setup & Build Instructions

Yes, you can run this on Linux! The Overgrowth engine and the RL module are fully cross-platform.

## 1. Dependencies
Install the required packages for your distribution.

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install build-essential cmake mesa-common-dev libsdl2-dev libsdl2-net-dev \
    libgtk2.0-dev libogg-dev libvorbis-dev libopenal-dev libjpeg-dev \
    libbz2-dev libfreetype-dev python3-dev python3-pip
```

### Fedora
```bash
sudo dnf install gcc-c++ cmake make 'pkgconfig(gl)' 'pkgconfig(glu)' \
    'pkgconfig(sdl2)' 'pkgconfig(SDL2_net)' 'pkgconfig(gtk+-2.0)' \
    'pkgconfig(ogg)' 'pkgconfig(vorbis)' 'pkgconfig(openal)' \
    'pkgconfig(libjpeg)' 'pkgconfig(bzip2)' 'pkgconfig(freetype2)' \
    python3-devel
```

## 2. Game Assets
You need the `Data` folder from the commercial version of Overgrowth.
- If you have Steam on Linux, install Overgrowth.
- Path is usually: `~/.steam/steam/steamapps/common/Overgrowth`

## 3. Building the RL Module (Standalone)
To build just the python module for testing:

```bash
cd /path/to/repo
pip install -e .
```

## 4. Building the Full Game (Integration)
To build the full game with the RL integration:

1.  Create a build directory:
    ```bash
    mkdir Build
    cd Build
    ```

2.  Configure CMake (replace path with your actual game path):
    ```bash
    cmake ../Projects -DAUX_DATA="~/.steam/steam/steamapps/common/Overgrowth" -DOG_RL=ON
    ```

3.  Compile:
    ```bash
    make -j$(nproc)
    ```

4.  Run:
    ```bash
    ./Overgrowth
    ```
