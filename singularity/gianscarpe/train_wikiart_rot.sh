#!/bin/bash
cd /home/gscarpellini/PuzzleDiffusion-GNN/singularity/gianscarpe

dataset='wikiart'
puzzles='6'
steps=300
gpus=3
cpus=6
batch_size=32
sampling='DDIM'
inference_ratio=10

NAME="Diff-${dataset}-${puzzles}-${steps}"
ARGS="-dataset $dataset -puzzle_sizes $puzzles -inference_ratio $inference_ratio -sampling $sampling -gpus $gpus -batch_size $batch_size -steps $steps -num_workers $cpus --noise_weight 0 --rotation True --checkpoint_path Puzzle-Diff/gru93lu3/checkpoints/epoch=189-step=101650.ckpt"

echo $NAME
echo ""
echo $ARGS
qsub -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_script.py,args="$ARGS" -N "$NAME"  pbs_args.sh
