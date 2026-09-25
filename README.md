## Setup
Python 3.9 was used.

```bash
conda create -n netmf python=3.9 -y
conda activate netmf
python -m pip install numpy==1.23.5 scipy==1.10.1 scikit-learn psutil matplotlib
```

The original implementation used Theano for element-wise operations. These
operations were replaced with equivalent NumPy operations for compatibility
with current Python/macOS environments.

## Files

```text
data/                       Benchmark datasets
embeddings/                 Generated embeddings and singular values
netmf.py                    NetMF + Task 2/3 instrumentation
predict.py                  Multi-label classification
prepare_blogcatalog.py      BlogCatalog preprocessing
prepare_flickr.py           Flickr preprocessing
plot_singular_values.py     Singular-value plot
singular_value_decay.png    Generated spectrum figure
hw1-MLWithGraphs.pdf        Final report
```

## Task 1 - Generate Embeddings

```bash
python netmf.py --input data/blogcatalog.mat --output embeddings/blogcatalog.npy
python netmf.py --input data/ppi.mat --output embeddings/ppi_test.npy
python netmf.py --input data/wikipedia.mat --output embeddings/wikipedia_test.npy
```

Run classification:

```bash
python predict.py --label data/blogcatalog.mat --embedding embeddings/blogcatalog.npy --seed 0
python predict.py --label data/ppi.mat --embedding embeddings/ppi_test.npy --seed 0
python predict.py --label data/wikipedia.mat --embedding embeddings/wikipedia_test.npy --seed 0
```

`predict.py` reports Micro-F1 and Macro-F1 for training ratios from 10% to 90%.

## Task 2 - Sparsity and Spectrum

Running `netmf.py` automatically reports:

- adjacency matrix density
- NetMF matrix density
- rank-128 reconstructed matrix density
- embedding density
- top singular values

Singular values are automatically saved under `embeddings/`.

Generate the spectrum plot with:

```bash
python plot_singular_values.py
```

Output:

```text
singular_value_decay.png
```

## Task 3 - Runtime and Memory

`netmf.py` automatically reports:

- eigen-decomposition time
- NetMF matrix construction time
- SVD factorization time
- memory usage
- peak RAM

Run Flickr separately for the scalability experiment:

```bash
python netmf.py --input data/flickr.mat --output embeddings/flickr.npy
```

On the machine used for this homework, Flickr was terminated during NetMF
matrix construction because of its memory requirement. This result is
documented in the report.

## Dataset Preparation

The submitted `.mat` files can be used directly.

To regenerate BlogCatalog or Flickr from the included raw data:

```bash
python prepare_blogcatalog.py
python prepare_flickr.py
```

## Report

See:

```text
hw1-MLWithGraphs.pdf
```

for experimental results, comparison with the NetMF paper, sparsity/spectrum
analysis, and scalability discussion.

## Reproducibility Note

Classification uses `--seed 0`. Runtime and memory measurements may vary
across machines.