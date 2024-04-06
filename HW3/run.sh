
batch_size=1
save_dir="test_bsz${batch_size}"


echo saving to ${save_dir}
python test.py \
    --save_dir ${save_dir} \
    --batch_size ${batch_size} \