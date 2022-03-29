root_dir=$(
    cd `dirname $0`;
    pwd
)/../
export PYTHONPATH=${root_dir}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

data_config=${root_dir}/configs/dataset.json
model_config=${root_dir}/configs/model.json
model_path=/home/data10/gaoshengyi/TreeRecognition/TreeSegmentation/checkpoints/best.pth
output_path=/home/data10/gaoshengyi/TreeRecognition/Output/synth
time=$(date "+%Y%m%d%H%M%S")

python ${root_dir}/synthesize.py ${data_config} ${model_config} ${model_path} ${output_path} \
    --work_num=10 \
    --use_gpu \
    # > ${root_dir}/logs/synth-${time}.log 2>&1
