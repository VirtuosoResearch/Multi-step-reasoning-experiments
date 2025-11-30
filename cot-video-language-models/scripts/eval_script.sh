# Use facebook/Perception-LM-1B for 1B parameters model and facebook/Perception-LM-8B for 8B parameters model.
CHECKPOINTS_PATH=facebook/Perception-LM-1B

# Define the tasks you want to evaluate PLM on. We support all the tasks present in lmms-eval, however have tested the following tasks with our models.

ALL_TASKS=(
    "docvqa" "chartqa" "textvqa" "infovqa" "ai2d_no_mask" "ok_vqa" "vizwiz_vqa" "mme"
    "realworldqa" "pope" "mmmu" "ocrbench" "coco_karpathy_val" "nocaps" "vqav2_val"
    "mvbench" "videomme" "vatex_test" "egoschema" "egoschema_subset" "mlvu_dev"
    "tempcompass_multi_choice" "perceptiontest_val_mc" "perceptiontest_test_mc"
)

# After specifying the task/tasks to evaluate, run the following command to start the evaluation.
SELECTED_TASK="textvqa,videomme"
accelerate launch --num_processes=8 \
-m lmms_eval \
--model plm \
--model_args pretrained=$CHECKPOINTS_PATH \
--tasks $SELECTED_TASK \
--batch_size 1 \
--log_samples \
--log_samples_suffix plm \
--output_path $OUTPUT_PATH