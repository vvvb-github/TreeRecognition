root_dir=$(
    cd `dirname $0`;
    pwd
)/../
export PYTHONPATH=${root_dir}

img_path=/home/data10/gaoshengyi/TreeRecognition/Output/skeleton
output_path=/home/data10/gaoshengyi/TreeRecognition/Output/keypoint
time=$(date "+%Y%m%d%H%M%S")

python ${root_dir}/test/keypoint.py ${img_path} ${output_path} \
    --radius=2 \
    # > ${root_dir}/logs/train-${time}.log 2>&1
