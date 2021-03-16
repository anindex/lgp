# lgp

This repo implements Dynamic Logic-Geometric Programming for 2D agents on PyBullet.

## Installation
This assumes you a;ready install the dependencies for Simon's master thesis repo and `humoro`.

Clone Simon's master thesis repo:

```bash
git clone https://animal.informatik.uni-stuttgart.de/simon.hagenmayer/masterthesis
```

Then, clone `humoro` and `lgp` to `masterthesis` folder, checkout `MASimon` branch on `humoro` and install dependencies of `lgp`:

```bash
cd masterthesis
git clone https://animal.informatik.uni-stuttgart.de/philippkratzer/humoro
git clone https://github.com/humans-to-robots-motion/lgp
cd humoro
git checkout MASimon
cd ../lgp
pip install -r requirements.txt
```

Finally, please download [MoGaze](https://humans-to-robots-motion.github.io/mogaze/) dataset and unzip it into `lgp/datasets/mogaze`.

## Usage

For now, there are two example scenarios `set_table1` and `set_table2` using the same PDDL domain `domain_set_table.pddl`, which reside in `lgp/data/scenarios`. 

They are tested on MoGaze dataset segment 1. This runs `set_table1`:

```bash
cd lgp
python3 examples/dynamic_lgp_humoro.py -p 1 -v True
```

Change to `-p 2` to run `set_table2`. And `-v` is verbose flag. 