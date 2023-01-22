# Animation Encoder in 2023

> **Note**
> Work in progress.


## Overview

anim_encoder creates small JavaScript + HTML animations from a series on PNG images.
This is a fork of the original work to do a few things:

- Update the script from Python 2 to Python 3
- Improve the usability of the script (more options)
- Modernize the JavaScript player to use new browser features.

Original details are at http://www.sublimetext.com/~jps/animated_gifs_the_hard_way.html

## Getting Started

Clone the repository and install the required dependencies with `pip`.

```
pip install -r requirements.txt
```

Alternatively, you may install it via your system's package manager.

> **Warning**
> This is deprecated and will not be updated in the future.

```sh
sudo apt-get install pngcrush python-opencv python-numpy python-scipy
```

Prepare the animation that you want to encode and convert them into image sequences.
For example, to encode image sequences in `capture/`:

```sh
python3 anim_encoder.py capture
```

This will generate `capture_packed.png` and `capture_anim.js`.

To specify a different output name, you can use `-o` or `--output`.

```sh
python3 anim_encoder.py -o output capture
```

In this case it will generate `output_packed.png` and `output_anim.js`.

If you have `pngcrush` or `pngquant` installed, you may be interested in
optimizing the output image further.
For animations that has fewer colors such as screen recordings, `pngquant`
provides significant compression. 

``` sh
# -C uses pngquant, -c uses pngcrush
python3 anim_encoder.py -C -c capture
```

For more help, run `python3 anim_encoder.py --help` to see all options.


## Capturing your own images

Images will be saved to `capture`, you simply need to run `capture.py` and then go about your task.
Note you can just delete frames you don't want as you initially set up, should save you some time. Then to run the program just go

```
python capture.py
```

If you need to change any settings it should be pretty simple just jump over to config.py
and edit the configuration options.


## Tips for producing smaller files

If you are not capturing the video via `capture.py`, then you need to use a
lossless video codec (`ffv1`, `huffyuv`, `lagarith`, etc.).
Many codecs such as h.264 produces quantization noise which completely messes
up the diffing algorithm and produces a big final result.

