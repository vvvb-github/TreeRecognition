root_dir=$(
    cd `dirname $0`;
    pwd
)/../
export PYTHONPATH=${root_dir}

img_path=/home/data10/gaoshengyi/TreeRecognition/Output/synth
output=/home/data10/gaoshengyi/TreeRecognition/Output/result.txt
store=/home/data10/gaoshengyi/TreeRecognition/Output/recognize
time=$(date "+%Y%m%d%H%M%S")

python ${root_dir}/main.py ${img_path} ${output} \
    --save=${store} \
    # --load=${store} \
    # > ${root_dir}/logs/train-${time}.log 2>&1
