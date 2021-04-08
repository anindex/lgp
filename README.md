# lgp

This repo implements Dynamic Logic-Geometric Programming for 2D agents on PyBullet.

## Installation
This assumes you a;ready install the dependencies for Simon's master thesis repo and `humoro`.

Clone Simon's master thesis repo:

```bash
git clone git@animal.informatik.uni-stuttgart.de:simon.hagenmayer/hierarchical-hmp.git
```

Then, clone `humoro` and `lgp` to `hierarchical-hmp` folder, checkout `MASimon` branch on `humoro` and install dependencies of `lgp`:

```bash
cd hierarchical-hmp
git clone git@animal.informatik.uni-stuttgart.de:philippkratzer/humoro.git
git clone https://github.com/humans-to-robots-motion/lgp
cd humoro
git checkout MASimon
cd ../lgp
pip install -r requirements.txt
```

Also clone `bewego` into `hierarchical-hmp` folder:
```bash
cd hierarchical-hmp
git clone https://github.com/anindex/bewego --recursive
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
make install
```

Finally, please download [MoGaze](https://humans-to-robots-motion.github.io/mogaze/) dataset and unzip it into `lgp/datasets/mogaze`.
```bash
mkdir -p datasets && cd datasets
wget https://ipvs.informatik.uni-stuttgart.de/mlr/philipp/mogaze/mogaze.zip
unzip mogaze.zip
```

And also run this script to initialize Pepper URDF:

```
cd lgp
python3 examples/init_pepper.py
```

## Usage

To see an example of Dynamic LGP, please run:

```
python3 examples/test_lgp.py -d True -v True -p False
```

where `-d` is the option to run in dynamic mode or not, `-v True` is to enable trajectory optimization visualization and `-p` is the option to enable human prediction.

To run Dynamic LGP with human prediction, please use the following segment `('p2_1', 137536, 139256)`:

```
python3 examples/test_lgp.py --segment "('p2_1', 137536, 139256)" -d True -v True -p True
```

To reproduce the paper results, please run the following experiment scripts.
- **Dynamic LGP with Human Ground Truth**:

```
python3 examples/experiment_ground_truth.py
```

- **Dynamic LGP with Long-term prediction**:

```
python3 examples/experiment_prediction.py
```

The experiment data will be saved in folder `lgp/data/experiments`. To visulize data result, please run:

```
python3 examples/process_data.py --name <your data>.p
```