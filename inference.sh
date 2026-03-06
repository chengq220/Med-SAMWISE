#!/usr/bin/env bash

# python inference.py \
#     --sam2_version 'med' \
#     --input_path data/endovis2018/valid/JPEGImages/seq1 \
#     --mask_input data/endovis2018/valid/VOS/seq1 \
#     --resume output/endovis2017_med/checkpoint0002.pth \
#     --text_prompts 'all' \
#     --num_frames 8 \
#     --threshold 0.5 \
#     --HSA 

# for seq in seq1 seq2 seq3 seq4 seq5 seq6 seq7 seq8 seq9 seq10; do
#     python inference.py \
#         --sam2_version 'med' \
#         --input_path "data/endovis2017/valid/JPEGImages/$seq" \
#         --mask_input "data/endovis2017/valid/VOS/$seq" \
#         --resume output/endovis2017_med/checkpoint0004.pth \
#         --text_prompts 'all' \
#         --num_frames 8 \
#         --threshold 0.5 \
#         --HSA
# done

for seq in seq2 seq5 seq9 seq15; do
    python inference.py \
        --sam2_version 'med' \
        --input_path "data/endovis2018/valid/JPEGImages/$seq" \
        --mask_input "data/endovis2018/valid/VOS/$seq" \
        --resume output/endovis2018_med/checkpoint0002.pth \
        --text_prompts 'all' \
        --num_frames 8 \
        --threshold 0.5 \
        --HSA
done
