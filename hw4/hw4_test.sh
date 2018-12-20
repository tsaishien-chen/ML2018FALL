wget 'https://www.dropbox.com/s/md9p54n5v766b1i/best_dnn.h5?dl=1'
wget 'https://www.dropbox.com/s/9l8mdomo1sj36z9/best_rnn.h5?dl=1'
wget 'https://www.dropbox.com/s/xub4qm4jxnpn3p6/word2vec_256_dnn.model?dl=1'
wget 'https://www.dropbox.com/s/tvggky45duq3dz5/word2vec_256_rnn.model?dl=1'
mv word2vec_256_rnn.model?dl=1 word2vec_256_rnn.model
mv word2vec_256_dnn.model?dl=1 word2vec_256_dnn.model
mv best_rnn.h5?dl=1 best_rnn.h5
mv best_dnn.h5?dl=1 best_dnn.h5
python3 test.py $1 $2 $3 word2vec_256_rnn.model best_rnn.h5 word2vec_256_dnn.model best_dnn.h5
