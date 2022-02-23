TFDS_DIR=/tmp/tensorflow_datasets/
DATA_DIR=/tmp/data/
DATASET=lm1b
SPLIT=test
THRESHOLD=100
CACHE=/tmp/cache/

cargo build

python3 scripts/load_dataset.py --data_dir $TFDS_DIR --save_dir $DATA_DIR --name $DATASET --split $SPLIT

python3 scripts/make_suffix_array.py $DATA_DIR$DATASET.$SPLIT

cargo run self-similar --data-file $DATA_DIR$DATASET.$SPLIT --length-threshold $THRESHOLD --cache-dir $CACHE --num-threads 96

cargo run collect --data-name $DATASET.$SPLIT --cache-dir $CACHE > /tmp/$DATASET.$SPLIT.remove.byterange


