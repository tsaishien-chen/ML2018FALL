python3 word2vec.py 256 $4 $1 $3
mkdir model_rnn_record
mkdir model_dnn_record
python3 train_rnn.py $1 $2 $4 word2vec_256.model
python3 train_dnn.py $1 $2 $4 word2vec_256.model
