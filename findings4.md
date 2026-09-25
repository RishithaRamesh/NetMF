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