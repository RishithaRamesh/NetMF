(netmf) rishitharamesh@Mac NetMF % curl -L -o data/wikipedia.mat \
http://snap.stanford.edu/node2vec/POS.mat
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 2264k  100 2264k    0     0   449k      0  0:00:05  0:00:05 --:--:--  486k
(netmf) rishitharamesh@Mac NetMF % ls -lh data/wikipedia.mat
-rw-r--r--@ 1 rishitharamesh  staff   2.2M Sep 24 22:58 data/wikipedia.mat
(netmf) rishitharamesh@Mac NetMF % python -c "import scipy.io; d=scipy.io.loadmat('data/wikipedia.mat'); print([(k, v.shape) for k,v in d.items() if not k.startswith('__')])"
[('group', (4777, 40)), ('network', (4777, 4777))]
(netmf) rishitharamesh@Mac NetMF % python netmf.py \
  --input data/wikipedia.mat \
  --output embeddings/wikipedia.npy
2026-09-24 22:59:50,230 Running NetMF for a large window size...
2026-09-24 22:59:50,230 Window size is set to be 10
2026-09-24 22:59:50,233 loading mat file data/wikipedia.mat
2026-09-24 22:59:50,243 Eigen decomposition...
2026-09-24 22:59:55,945 Maximum eigenvalue 1.000000, minimum eigenvalue 0.211298
2026-09-24 22:59:55,945 Computing D^{-1/2}U..
2026-09-24 22:59:55,950 After filtering, max eigenvalue=1.000000, min eigenvalue=0.026791
2026-09-24 22:59:56,371 Computed DeepWalk matrix with 9539005 non-zero elements
2026-09-24 23:00:03,334 Save embedding to embeddings/wikipedia.npy



(netmf) rishitharamesh@Mac NetMF % python predict.py \
  --label data/wikipedia.mat \
  --embedding embeddings/wikipedia.npy \
  --seed 0 2>&1 | grep "Average"
2026-09-24 23:13:43,766 Average micro 46.51, Average macro 8.58
2026-09-24 23:13:45,463 Average micro 48.77, Average macro 9.82
2026-09-24 23:13:47,900 Average micro 49.98, Average macro 10.24
2026-09-24 23:13:50,596 Average micro 50.67, Average macro 10.32
2026-09-24 23:13:53,581 Average micro 51.20, Average macro 10.51
2026-09-24 23:13:57,332 Average micro 51.37, Average macro 10.73
2026-09-24 23:14:01,619 Average micro 51.56, Average macro 10.76
2026-09-24 23:14:06,179 Average micro 51.89, Average macro 10.80
2026-09-24 23:14:11,139 Average micro 52.59, Average macro 11.16