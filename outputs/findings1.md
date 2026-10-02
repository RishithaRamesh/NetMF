(netmf) rishitharamesh@Mac NetMF % python netmf.py \
  --input data/blogcatalog.mat \
  --output embeddings/blogcatalog.npy
2026-09-24 22:33:19,325 Running NetMF for a large window size...
2026-09-24 22:33:19,325 Window size is set to be 10
2026-09-24 22:33:19,329 loading mat file data/blogcatalog.mat
2026-09-24 22:33:19,366 Eigen decomposition...
2026-09-24 22:33:34,379 Maximum eigenvalue 1.000000, minimum eigenvalue 0.192003
2026-09-24 22:33:34,379 Computing D^{-1/2}U..
2026-09-24 22:33:34,384 After filtering, max eigenvalue=1.000000, min eigenvalue=0.023763
2026-09-24 22:33:36,196 Computed DeepWalk matrix with 43359474 non-zero elements
2026-09-24 22:34:04,818 Save embedding to embeddings/blogcatalog.npy
(netmf) rishitharamesh@Mac NetMF % python -c "import numpy as np; x=np.load('embeddings/blogcatalog.npy'); print(x.shape)"
(10312, 128)
(netmf) rishitharamesh@Mac NetMF % python predict.py \
  --label data/blogcatalog.mat \
  --embedding embeddings/blogcatalog.npy \
  --seed 0 \
  --start-train-ratio 10 \
  --stop-train-ratio 10 \
  --num-train-ratio 1
2026-09-24 22:47:42,190 Loading label from data/blogcatalog.mat...
2026-09-24 22:47:42,194 loading mat file data/blogcatalog.mat
(10312, 39) <class 'numpy.ndarray'> 0 1
2026-09-24 22:47:42,196 Label loaded!
2026-09-24 22:47:42,196 Loading network embedding from embeddings/blogcatalog.npy...
2026-09-24 22:47:42,202 Network embedding loaded!
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:47:43,492 micro f1 0.382565 macro f1 0.224931
(1031,) (9281,)
2026-09-24 22:47:43,784 micro f1 0.381668 macro f1 0.220765
(1031,) (9281,)
2026-09-24 22:47:44,045 micro f1 0.380594 macro f1 0.231362
(1031,) (9281,)
2026-09-24 22:47:44,366 micro f1 0.381512 macro f1 0.229519
(1031,) (9281,)
2026-09-24 22:47:44,651 micro f1 0.384953 macro f1 0.231454
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 37 is present in all training examples.
  warnings.warn(
2026-09-24 22:47:44,915 micro f1 0.385892 macro f1 0.232351
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 37 is present in all training examples.
  warnings.warn(
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:47:45,200 micro f1 0.386402 macro f1 0.232893
(1031,) (9281,)
2026-09-24 22:47:45,492 micro f1 0.387414 macro f1 0.231017
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:47:45,653 micro f1 0.381544 macro f1 0.230053
(1031,) (9281,)
2026-09-24 22:47:45,898 micro f1 0.383313 macro f1 0.225804
2026-09-24 22:47:45,898 10 fold validation, training ratio 0.100000
2026-09-24 22:47:45,898 Average micro 38.36, Average macro 22.90

Micro-F1: 38.36%
Macro-F1: 22.90%

-----------------------------------------------------------------------------------------------------------
(netmf) rishitharamesh@Mac NetMF % python predict.py \
  --label data/blogcatalog.mat \
  --embedding embeddings/blogcatalog.npy \
  --seed 0
2026-09-24 22:51:45,224 Loading label from data/blogcatalog.mat...
2026-09-24 22:51:45,225 loading mat file data/blogcatalog.mat
(10312, 39) <class 'numpy.ndarray'> 0 1
2026-09-24 22:51:45,226 Label loaded!
2026-09-24 22:51:45,226 Loading network embedding from embeddings/blogcatalog.npy...
2026-09-24 22:51:45,228 Network embedding loaded!
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:46,499 micro f1 0.382565 macro f1 0.224931
(1031,) (9281,)
2026-09-24 22:51:46,798 micro f1 0.381668 macro f1 0.220765
(1031,) (9281,)
2026-09-24 22:51:47,108 micro f1 0.380594 macro f1 0.231362
(1031,) (9281,)
2026-09-24 22:51:47,425 micro f1 0.381512 macro f1 0.229519
(1031,) (9281,)
2026-09-24 22:51:47,710 micro f1 0.384953 macro f1 0.231454
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 37 is present in all training examples.
  warnings.warn(
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:48,045 micro f1 0.385892 macro f1 0.232351
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 37 is present in all training examples.
  warnings.warn(
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:48,339 micro f1 0.386402 macro f1 0.232893
(1031,) (9281,)
2026-09-24 22:51:48,681 micro f1 0.387414 macro f1 0.231017
(1031,) (9281,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:48,968 micro f1 0.381544 macro f1 0.230053
(1031,) (9281,)
2026-09-24 22:51:49,272 micro f1 0.383313 macro f1 0.225804
2026-09-24 22:51:49,272 10 fold validation, training ratio 0.100000
2026-09-24 22:51:49,272 Average micro 38.36, Average macro 22.90
(2062,) (8250,)
2026-09-24 22:51:49,696 micro f1 0.402910 macro f1 0.252197
(2062,) (8250,)
2026-09-24 22:51:50,083 micro f1 0.403433 macro f1 0.255174
(2062,) (8250,)
2026-09-24 22:51:50,433 micro f1 0.404476 macro f1 0.255081
(2062,) (8250,)
2026-09-24 22:51:50,851 micro f1 0.404351 macro f1 0.249179
(2062,) (8250,)
2026-09-24 22:51:51,226 micro f1 0.401125 macro f1 0.255789
(2062,) (8250,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:51,540 micro f1 0.405352 macro f1 0.255477
(2062,) (8250,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:51,879 micro f1 0.407733 macro f1 0.258218
(2062,) (8250,)
2026-09-24 22:51:52,258 micro f1 0.405266 macro f1 0.247395
(2062,) (8250,)
2026-09-24 22:51:52,604 micro f1 0.404522 macro f1 0.252016
(2062,) (8250,)
2026-09-24 22:51:52,942 micro f1 0.407943 macro f1 0.261791
2026-09-24 22:51:52,942 10 fold validation, training ratio 0.200000
2026-09-24 22:51:52,942 Average micro 40.47, Average macro 25.42
(3093,) (7219,)
2026-09-24 22:51:53,366 micro f1 0.415905 macro f1 0.270174
(3093,) (7219,)
2026-09-24 22:51:53,808 micro f1 0.413653 macro f1 0.268409
(3093,) (7219,)
2026-09-24 22:51:54,295 micro f1 0.413605 macro f1 0.268073
(3093,) (7219,)
2026-09-24 22:51:54,751 micro f1 0.412662 macro f1 0.258802
(3093,) (7219,)
2026-09-24 22:51:55,229 micro f1 0.414345 macro f1 0.270186
(3093,) (7219,)
/opt/anaconda3/envs/netmf/lib/python3.9/site-packages/sklearn/multiclass.py:90: UserWarning: Label not 38 is present in all training examples.
  warnings.warn(
2026-09-24 22:51:55,716 micro f1 0.415508 macro f1 0.268563
(3093,) (7219,)
2026-09-24 22:51:56,161 micro f1 0.418332 macro f1 0.268236
(3093,) (7219,)
2026-09-24 22:51:56,634 micro f1 0.414231 macro f1 0.265319
(3093,) (7219,)
2026-09-24 22:51:57,093 micro f1 0.423461 macro f1 0.275473
(3093,) (7219,)
2026-09-24 22:51:57,600 micro f1 0.421615 macro f1 0.276189
2026-09-24 22:51:57,600 10 fold validation, training ratio 0.300000
2026-09-24 22:51:57,600 Average micro 41.63, Average macro 26.89
(4124,) (6188,)
2026-09-24 22:51:58,169 micro f1 0.424306 macro f1 0.277014
(4124,) (6188,)
2026-09-24 22:51:58,764 micro f1 0.421047 macro f1 0.277531
(4124,) (6188,)
2026-09-24 22:51:59,321 micro f1 0.417497 macro f1 0.271762
(4124,) (6188,)
2026-09-24 22:51:59,947 micro f1 0.418194 macro f1 0.266416
(4124,) (6188,)
2026-09-24 22:52:00,518 micro f1 0.423175 macro f1 0.281506
(4124,) (6188,)
2026-09-24 22:52:01,092 micro f1 0.423152 macro f1 0.269754
(4124,) (6188,)
2026-09-24 22:52:01,683 micro f1 0.425845 macro f1 0.276711
(4124,) (6188,)
2026-09-24 22:52:02,215 micro f1 0.423675 macro f1 0.278960
(4124,) (6188,)
2026-09-24 22:52:02,788 micro f1 0.424648 macro f1 0.278741
(4124,) (6188,)
2026-09-24 22:52:03,222 micro f1 0.422840 macro f1 0.280510
2026-09-24 22:52:03,222 10 fold validation, training ratio 0.400000
2026-09-24 22:52:03,222 Average micro 42.24, Average macro 27.59
(5156,) (5156,)
2026-09-24 22:52:03,843 micro f1 0.428991 macro f1 0.280206
(5156,) (5156,)
2026-09-24 22:52:04,582 micro f1 0.429028 macro f1 0.284911
(5156,) (5156,)
2026-09-24 22:52:05,228 micro f1 0.424990 macro f1 0.276376
(5156,) (5156,)
2026-09-24 22:52:05,860 micro f1 0.424100 macro f1 0.273723
(5156,) (5156,)
2026-09-24 22:52:06,530 micro f1 0.428651 macro f1 0.286307
(5156,) (5156,)
2026-09-24 22:52:07,181 micro f1 0.428196 macro f1 0.274736
(5156,) (5156,)
2026-09-24 22:52:07,855 micro f1 0.429358 macro f1 0.281609
(5156,) (5156,)
2026-09-24 22:52:08,503 micro f1 0.422394 macro f1 0.280083
(5156,) (5156,)
2026-09-24 22:52:09,167 micro f1 0.431740 macro f1 0.287138
(5156,) (5156,)
2026-09-24 22:52:09,822 micro f1 0.429238 macro f1 0.280075
2026-09-24 22:52:09,822 10 fold validation, training ratio 0.500000
2026-09-24 22:52:09,822 Average micro 42.77, Average macro 28.05
(6187,) (4125,)
2026-09-24 22:52:10,537 micro f1 0.433729 macro f1 0.287526
(6187,) (4125,)
2026-09-24 22:52:11,228 micro f1 0.434813 macro f1 0.287073
(6187,) (4125,)
2026-09-24 22:52:11,956 micro f1 0.427350 macro f1 0.275310
(6187,) (4125,)
2026-09-24 22:52:12,745 micro f1 0.425181 macro f1 0.277833
(6187,) (4125,)
2026-09-24 22:52:13,426 micro f1 0.432574 macro f1 0.291924
(6187,) (4125,)
2026-09-24 22:52:14,166 micro f1 0.430825 macro f1 0.281268
(6187,) (4125,)
2026-09-24 22:52:14,901 micro f1 0.430703 macro f1 0.289757
(6187,) (4125,)
2026-09-24 22:52:15,618 micro f1 0.426756 macro f1 0.283085
(6187,) (4125,)
2026-09-24 22:52:16,302 micro f1 0.429905 macro f1 0.290631
(6187,) (4125,)
2026-09-24 22:52:17,009 micro f1 0.431624 macro f1 0.282570
2026-09-24 22:52:17,010 10 fold validation, training ratio 0.600000
2026-09-24 22:52:17,010 Average micro 43.03, Average macro 28.47
(7218,) (3094,)
2026-09-24 22:52:17,795 micro f1 0.433419 macro f1 0.286562
(7218,) (3094,)
2026-09-24 22:52:18,625 micro f1 0.443023 macro f1 0.294391
(7218,) (3094,)
2026-09-24 22:52:19,384 micro f1 0.429951 macro f1 0.269981
(7218,) (3094,)
2026-09-24 22:52:20,230 micro f1 0.417469 macro f1 0.268910
(7218,) (3094,)
2026-09-24 22:52:21,111 micro f1 0.438678 macro f1 0.290947
(7218,) (3094,)
2026-09-24 22:52:21,925 micro f1 0.428737 macro f1 0.282968
(7218,) (3094,)
2026-09-24 22:52:22,768 micro f1 0.436389 macro f1 0.289817
(7218,) (3094,)
2026-09-24 22:52:23,648 micro f1 0.428439 macro f1 0.288813
(7218,) (3094,)
2026-09-24 22:52:24,490 micro f1 0.435753 macro f1 0.294255
(7218,) (3094,)
2026-09-24 22:52:25,321 micro f1 0.436066 macro f1 0.282509
2026-09-24 22:52:25,321 10 fold validation, training ratio 0.700000
2026-09-24 22:52:25,321 Average micro 43.28, Average macro 28.49
(8249,) (2063,)
2026-09-24 22:52:26,294 micro f1 0.432090 macro f1 0.280244
(8249,) (2063,)
2026-09-24 22:52:27,164 micro f1 0.442334 macro f1 0.303895
(8249,) (2063,)
2026-09-24 22:52:28,056 micro f1 0.436818 macro f1 0.272419
(8249,) (2063,)
2026-09-24 22:52:28,951 micro f1 0.414222 macro f1 0.273351
(8249,) (2063,)
2026-09-24 22:52:29,822 micro f1 0.429420 macro f1 0.284437
(8249,) (2063,)
2026-09-24 22:52:30,736 micro f1 0.436794 macro f1 0.281151
(8249,) (2063,)
2026-09-24 22:52:31,635 micro f1 0.447782 macro f1 0.293275
(8249,) (2063,)
2026-09-24 22:52:32,482 micro f1 0.431802 macro f1 0.287511
(8249,) (2063,)
2026-09-24 22:52:33,350 micro f1 0.437020 macro f1 0.297517
(8249,) (2063,)
2026-09-24 22:52:34,219 micro f1 0.445840 macro f1 0.298770
2026-09-24 22:52:34,219 10 fold validation, training ratio 0.800000
2026-09-24 22:52:34,219 Average micro 43.54, Average macro 28.73
(9280,) (1032,)
2026-09-24 22:52:35,229 micro f1 0.424773 macro f1 0.268038
(9280,) (1032,)
2026-09-24 22:52:36,215 micro f1 0.447059 macro f1 0.313128
(9280,) (1032,)
2026-09-24 22:52:37,136 micro f1 0.463516 macro f1 0.289537
(9280,) (1032,)
2026-09-24 22:52:38,118 micro f1 0.428862 macro f1 0.294043
(9280,) (1032,)
2026-09-24 22:52:39,112 micro f1 0.431497 macro f1 0.277499
(9280,) (1032,)
2026-09-24 22:52:40,038 micro f1 0.434097 macro f1 0.283596
(9280,) (1032,)
2026-09-24 22:52:41,028 micro f1 0.456867 macro f1 0.303324
(9280,) (1032,)
2026-09-24 22:52:41,988 micro f1 0.437543 macro f1 0.298068
(9280,) (1032,)
2026-09-24 22:52:42,938 micro f1 0.431321 macro f1 0.285903
(9280,) (1032,)
2026-09-24 22:52:43,928 micro f1 0.455432 macro f1 0.307952
2026-09-24 22:52:43,928 10 fold validation, training ratio 0.900000
2026-09-24 22:52:43,928 Average micro 44.11, Average macro 29.21

2026-09-24 22:52:43,928 10 fold validation, training ratio 0.900000
2026-09-24 22:52:43,928 Average micro 44.11, Average macro 29.21


--------------------------------------------------------------------------------------------------------------------------

new code fpr part 2 and 3:
netmf) rishitharamesh@Mac NetMF % python netmf.py \
  --input data/blogcatalog.mat \
  --output embeddings/blogcatalog_test.npy
2026-09-25 12:07:36,731 Running NetMF for a large window size...
2026-09-25 12:07:36,731 Window size is set to be 10
2026-09-25 12:07:36,733 loading mat file data/blogcatalog.mat
2026-09-25 12:07:36,733 Adjacency matrix: shape=(10312, 10312), nnz=667966, percent nonzero=0.628157%
2026-09-25 12:07:36,733 Memory after loading adjacency: 58.19 MB
2026-09-25 12:07:36,748 Eigen decomposition...
2026-09-25 12:07:49,277 Maximum eigenvalue 1.000000, minimum eigenvalue 0.192003
2026-09-25 12:07:49,277 Computing D^{-1/2}U..
2026-09-25 12:07:49,282 After filtering, max eigenvalue=1.000000, min eigenvalue=0.023763
2026-09-25 12:07:50,404 DeepWalk matrix: shape=(10312, 10312), nnz=43359474, percent nonzero=40.775397%
2026-09-25 12:07:50,404 Memory after DeepWalk matrix construction: 1868.84 MB
2026-09-25 12:07:51,491 DeepWalk matrix construction time: 2.21 seconds
2026-09-25 12:08:17,167 Top 10 singular values: [2136.16  ,  937.3992,  847.0129,  638.561 ,  525.2427,  513.0866,
  490.3676,  451.9428,  446.458 ,  416.8263]
2026-09-25 12:08:17,172 SVD factorization time: 25.68 seconds
2026-09-25 12:08:17,172 Memory after SVD: 786.02 MB
2026-09-25 12:08:17,172 Peak memory usage: 3192.09 MB
2026-09-25 12:08:17,173 Save embedding to embeddings/blogcatalog_test.npy