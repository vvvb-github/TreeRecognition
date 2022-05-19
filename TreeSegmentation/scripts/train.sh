root_dir=$(
    cd `dirname $0`;
    pwd
)/../
export PYTHONPATH=${root_dir}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_config=${root_dir}/configs/dataset.json
model_config=${root_dir}/configs/model.json
process_config=${root_dir}/configs/process.json
output_path=/home/data10/gaoshengyi/TreeRecognition/Output
time=$(date "+%Y%m%d%H%M%S")

python ${root_dir}/train.py ${data_config} ${model_config} ${process_config} ${output_path} \
    HRNetV2 \
    --work_num=10 \
    --use_gpu \
    --amp \
    # > ${root_dir}/logs/train-${time}.log 2>&1
