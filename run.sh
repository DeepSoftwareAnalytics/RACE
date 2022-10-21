lang=$1 

# optimizer
lr=5e-5
batch_size=32
beam_size=10
epochs=10

# model 
source_length=200
target_length=30

# data
data_dir=dataset/$lang/contextual_medits
train_file=$data_dir/train.jsonl
dev_file=$data_dir/valid.jsonl
test_file=$data_dir/test.jsonl


pretrained_model=Salesforce/codet5-base 

# ============ Step 1 Training ==============

function train_codet5 () {

output_dir=saved_model/codet5/${lang}/
mkdir -p $output_dir
echo $output_dir
echo "============TRAINING============"
 CUDA_VISIBLE_DEVICES=0 python run.py  --do_train --do_eval   --do_test --eval_frequency 100 \
  --run_codet5 \
  --model_name_or_path $pretrained_model \
  --train_filename $train_file \
  --dev_filename $dev_file \
  --test_filename ${test_file} \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --do_lower_case \
  --beam_size $beam_size --train_batch_size $batch_size \
  --eval_batch_size $batch_size --learning_rate $lr \
  --num_train_epochs $epochs --seed 0 2>&1|tee  $output_dir/train.log
}


# 
train_codet5


# ============ Step 2 Retrieval ==============

retrieval_result_dir=${data_dir}/codet5_retrieval_result
mkdir -p ${retrieval_result_dir}

function retrieval () {
echo "============retrieval ============"
retrieval_filename=$1 
load_model_path=saved_model/codet5/${lang}/checkpoint-best-bleu/pytorch_model.bin
 CUDA_VISIBLE_DEVICES=0  python run.py   --do_retrieval \
 --run_codet5 \
 --is_cosine_space \
 --train_filename ${train_file} \
 --max_source_length $source_length \
 --max_target_length $target_length \
 --train_batch_size $batch_size \
 --eval_batch_size $batch_size \
 --retrieval_filename ${data_dir}/${retrieval_filename}.jsonl \
 --retrieval_result_dir ${retrieval_result_dir} \
 --retrieval_result_filename ${retrieval_filename}.jsonl \
 --load_model_path ${load_model_path} 2>&1 |tee ${retrieval_result_dir}/${retrieval_filename}.log.txt 
}




retrieval "train" 
retrieval "valid" 
retrieval "test" 

 # ============ Step 3 Refine ===============

train_retireved_file=${retrieval_result_dir}/train.jsonl
dev_retireved_file=${retrieval_result_dir}/valid.jsonl
test_retireved_file=${retrieval_result_dir}/test.jsonl

function refine () {
# --debug 
load_model_path=saved_model/codet5/${lang}/checkpoint-best-bleu/pytorch_model.bin
output_dir=saved_model/ECMG/${lang}/
mkdir -p $output_dir
echo $output_dir

echo "============Refining============"

  CUDA_VISIBLE_DEVICES=0 python run.py  --do_train --do_eval  --do_test --eval_frequency 100 \
  --load_finetuned_model_path ${load_model_path} \
  --model_name_or_path $pretrained_model \
  --train_filename $train_file \
  --dev_filename $dev_file \
  --test_filename ${test_file} \
  --train_retireved_filename $train_retireved_file \
  --dev_retireved_filename $dev_retireved_file \
  --test_retireved_filename ${test_retireved_file} \
  --output_dir $output_dir \
  --max_source_length $source_length \
  --max_target_length $target_length \
  --do_lower_case \
  --beam_size $beam_size --train_batch_size $batch_size \
  --eval_batch_size $batch_size --learning_rate $lr \
  --num_train_epochs $epochs --seed 0 2>&1|tee  $output_dir/refine.log
}


refine 