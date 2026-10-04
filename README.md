## Setup

Python 3.9 was used.

```bash
conda create -n netmf python=3.9 -y
conda activate netmf
python -m pip install numpy==1.23.5 scipy==1.10.1 scikit-learn psutil matplotlib
```

The original NetMF implementation used Theano for some element-wise operations.
These were replaced with equivalent NumPy operations for compatibility with the
current Python/macOS environment. The NetMF computation itself was otherwise
kept unchanged for the completed runs.

## Files

```text
data/                       Benchmark datasets and raw dataset files
embeddings/                 Generated embeddings and singular values
outputs/                    Final rerun logs used to verify reported results
netmf.py                    NetMF + Task 2/3 instrumentation
predict.py                  Multi-label classification
prepare_blogcatalog.py      BlogCatalog preprocessing
prepare_flickr.py           Flickr preprocessing
plot_singular_values.py     Singular-value plot generation
singular_value_decay.png    Generated spectrum figure
hw1-MLWithGraphs.pdf        Final report
```

## Task 1 - Generate Embeddings and Run Classification

Generate the embeddings used for the final reruns:

```bash
python netmf.py \
  --input data/blogcatalog.mat \
  --output embeddings/blogcatalog_rerun2.npy \
  2>&1 | tee outputs/rerun_blogcatalog_netmf_final.txt

python netmf.py \
  --input data/ppi.mat \
  --output embeddings/ppi_rerun.npy \
  2>&1 | tee outputs/rerun_ppi_netmf_final.txt

python netmf.py \
  --input data/wikipedia.mat \
  --output embeddings/wikipedia_rerun.npy \
  2>&1 | tee outputs/rerun_wikipedia_netmf_final.txt
```

Run multi-label classification:

```bash
python predict.py \
  --label data/blogcatalog.mat \
  --embedding embeddings/blogcatalog_rerun2.npy \
  --seed 0 \
  2>&1 | tee outputs/rerun_blogcatalog_predict_final.txt

python predict.py \
  --label data/ppi.mat \
  --embedding embeddings/ppi_rerun.npy \
  --seed 0 \
  2>&1 | tee outputs/rerun_ppi_predict_final.txt

python predict.py \
  --label data/wikipedia.mat \
  --embedding embeddings/wikipedia_rerun.npy \
  --seed 0 \
  2>&1 | tee outputs/rerun_wikipedia_predict_final.txt
```

`predict.py` reports Micro-F1 and Macro-F1 for training ratios from 10% to 90%
using 10 repeated random train/test splits.

The final reported classification values were checked against these rerun logs.

## Task 2 - Sparsity and Singular-Value Analysis

Running `netmf.py` automatically reports:

- adjacency-matrix nonzero percentage
- NetMF/DeepWalk-matrix nonzero percentage
- rank-128 reconstructed-matrix nonzero percentage
- embedding nonzero percentage
- leading singular values

Singular values are automatically saved under `embeddings/`.

Generate the singular-value decay plot with:

```bash
python plot_singular_values.py
```

Output:

```text
singular_value_decay.png
```

## Task 3 - Runtime and Memory

`netmf.py` automatically reports:

- eigendecomposition time
- NetMF matrix-construction time
- SVD factorization time
- current memory usage
- peak RAM usage

The SVD timer measures only the `scipy.sparse.linalg.svds()` call; post-SVD
reconstruction and density diagnostics are excluded.

### Flickr scalability run

Flickr did not complete NetMF matrix construction on the machine used for this
homework. The final monitored run used macOS `/usr/bin/time -l` so that elapsed
time and maximum resident set size were retained even when the process
terminated:

```bash
/usr/bin/time -l python netmf.py \
  --input data/flickr.mat \
  --output embeddings/flickr.npy \
  > outputs/rerun_flickr_netmf_final.txt 2>&1
```

In the final run, eigendecomposition completed, but NetMF matrix construction
did not finish and SVD/classification were not reached. The report discusses
this as a scalability limitation.

## Dataset Preparation

The submitted `.mat` files can be used directly.

To regenerate BlogCatalog or Flickr from the included raw CSV data:

```bash
python prepare_blogcatalog.py
python prepare_flickr.py
```

Expected MATLAB keys:

```text
network    sparse adjacency matrix
group      sparse multi-label matrix
```

## Report

See:

```text
hw1-MLWithGraphs.pdf
```

for:

- reproduction comparison with the NetMF paper
- Micro-F1 and Macro-F1 results
- matrix-density comparisons
- singular-value spectrum analysis
- runtime and peak-memory measurements
- Flickr scalability discussion
- closed-form factorization vs. stochastic training discussion

## Reproducibility Notes

- Classification uses `--seed 0`.
- Final NetMF and classification rerun logs are stored in `outputs/`.
- Runtime and memory measurements may vary across machines and operating systems.
- Sparse eigensolver/SVD outputs can differ by sign or basis orientation across
  runs even when singular values and downstream classification results are
  equivalent.
