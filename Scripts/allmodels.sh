#! /bin/sh

for model in p r s ss pss fs rs
do
    python3 Python/datagen.py -x 10 -y 50 --lens $model --name $model
done

