root_dir=$(
    cd `dirname $0`;
    pwd
)/../
export PYTHONPATH=${root_dir}

img_path=/home/data10/gaoshengyi/TreeRecognition/Output/synth
output_path=/home/data10/gaoshengyi/TreeRecognition/Output/skeleton
time=$(date "+%Y%m%d%H%M%S")

python ${root_dir}/test/skeleton.py ${img_path} ${output_path} \
    # > ${root_dir}/logs/skeleton-${time}.log 2>&1
