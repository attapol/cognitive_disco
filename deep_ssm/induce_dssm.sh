python arg_dump.py $1/pdtb-data.json > tmp_arg_dump.txt
SENT2VEC=~/nlp/lib/sent2vec/sample
/opt/mono/bin/mono $SENT2VEC/bin/sent2vec.exe --dssm_vec tmp_arg_dump.txt $SENT2VEC/dssm_model tmp_dssm_vec.out
/opt/mono/bin/mono $SENT2VEC/bin/sent2vec.exe --cdssm_vec tmp_arg_dump.txt $SENT2VEC/cdssm_model 20 tmp_cdssm_vec.out
python insert_vec.py $1/pdtb-data.json > $1/pdtb-data-plus.json
