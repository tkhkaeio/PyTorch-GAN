mkdir log
mkdir log/progress
export PYTHONUNBUFFERED="True"
LOG="log/progress/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python train.py \
--train_dir /content/drive/My\ Drive/data \
--save_colab_dir /content/drive/My\ Drive/DCGAN \
--config config.yml > $LOG \
--load_model False \
--restart 0