(netmf) rishitharamesh@Mac NetMF % python netmf.py \
  --input data/flickr.mat \
  --output embeddings/flickr.npy
2026-09-25 10:30:37,780 Running NetMF for a large window size...
2026-09-25 10:30:37,780 Window size is set to be 10
2026-09-25 10:30:37,810 loading mat file data/flickr.mat
2026-09-25 10:30:38,162 Eigen decomposition...
2026-09-25 10:31:47,373 Maximum eigenvalue 1.000000, minimum eigenvalue 0.375625
2026-09-25 10:31:47,373 Computing D^{-1/2}U..
2026-09-25 10:31:47,470 After filtering, max eigenvalue=1.000000, min eigenvalue=0.060157
zsh: killed     python netmf.py --input data/flickr.mat --output embeddings/flickr.npy


new code:
(netmf) rishitharamesh@Mac NetMF % python netmf.py \
  --input data/blogcatalog.mat \
  --output embeddings/blogcatalog_test.npy
2026-09-25 11:48:42,883 Running NetMF for a large window size...
2026-09-25 11:48:42,884 Window size is set to be 10
2026-09-25 11:48:42,888 loading mat file data/blogcatalog.mat
2026-09-25 11:48:42,888 Adjacency matrix: shape=(10312, 10312), nnz=667966, percent nonzero=0.628157%
2026-09-25 11:48:42,888 Memory after loading adjacency: 56.83 MB
2026-09-25 11:48:42,905 Eigen decomposition...