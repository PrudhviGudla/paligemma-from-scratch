@echo off

set MODEL_PATH=./paligemma-3b-mix-224
set PROMPT=detect car\n
set IMAGE_FILE_PATH=./test_images/bowie.jpg
set MAX_TOKENS_TO_GENERATE=128
set TEMPERATURE=0.8
set TOP_P=0.9
set DO_SAMPLE=False
set ONLY_CPU=False
set vae_checkpoint=vae-oid.npz
set output_path=./output_results/result2.png

python src/inference.py ^
    --model_path "%MODEL_PATH%" ^
    --prompt "%PROMPT%" ^
    --image_file_path "%IMAGE_FILE_PATH%" ^
    --max_tokens_to_generate %MAX_TOKENS_TO_GENERATE% ^
    --temperature %TEMPERATURE% ^
    --top_p %TOP_P% ^
    --do_sample %DO_SAMPLE% ^
    --only_cpu %ONLY_CPU% ^
    --vae_checkpoint "%vae_checkpoint%" ^
    --output_path "%output_path%"

pause
