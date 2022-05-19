root_dir=$(
    cd `dirname $0`;
    pwd
)/../
export PYTHONPATH=${root_dir}

img_path=/home/data10/gaoshengyi/TreeRecognition/Output/skeleton
output_path=/home/data10/gaoshengyi/TreeRecognition/Output/keypoint
time=$(date "+%Y%m%d%H%M%S")

# python ${root_dir}/test/keypoint.py ${img_path} ${output_path} \
#     --radius=2 \
    # > ${root_dir}/logs/train-${time}.log 2>&1

max_size=21
for ((i=3;i<=max_size;i+=2))
do
# echo ${output_path}/branch_kernel_${i}
python ${root_dir}/test/keypoint.py ${img_path} ${output_path}/branch_kernel_${i} --branch_n=${i} --radius=2
done
