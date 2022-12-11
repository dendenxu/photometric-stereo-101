# Computer Vision Homework #1 Report

## Homework Setting

> Given linearized render of an object from one view with multiple light direction and light intensity, recover the normal and albedo image of the object using a) a simple linear Lambertian model and b) a simple linear Lambertian model with highlight and shadow removal.

### Data specification

- 96 images of an object with a resolution of 612 * 512, stored as 16-bit `png`
- A mask file for valid object pixels
- Light directions and light intensities used for all rendered objects in `txt`

### Code structure

This code is implemented in `python`, powered by `pytorch`, other required packages are listed in `requirements.txt`

Assume you've got a valid `python` installation, run:

```shell
pip install -r requirements.txt # to install dependencies
```

We actually recommend installing `pytorch` with `cuda` support through `conda` (possibly with the help of `mamba`), but that hustle is beyond the scope of this small project... (just like the hustle of installing `matlab`, luckily we don't need `matlab`)

- `main.py`: for the main logic of the algorithm implemented in this project
- `utils.py`: for utility functions used by the main program
- `run_exps.py`: a script for running all experiment settings on all provided datasets (given a `--data_root` directory)

Both `main.py` and `run_exps.py` provides command line interfaces, run them with `--help` to see help to the command line arguments. Optionally, see `run_exps.py` for example use cases of `main.py`.

## Experiment Results

