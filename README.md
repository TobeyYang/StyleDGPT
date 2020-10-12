# StyleDGPT
This repo contains the code of the paper: [STYLEDGPT: Stylized Response Generation with Pre-trained Language Models](https://arxiv.org/abs/2010.02569), Findings of EMNLP2020. 


## Requirments
This code is tested on Python 3.6, Pytorch 1.3.0, and transformers 2.5.1.
```
pip install -r requirments.txt
```

## Resources

### Data
First, you need to prepare the data following the pipeline in this prior [work](https://github.com/golsun/StyleFusion). 
Of course, you can test our model on your custom corpora. The data format of the non-stylized dialogue corpus is TSV. 
It contains one piece of data per line, and the context and response is delimited by `\t`. 
```
scottish advent calendar        i would be so much more excited for this than chocolate !
scottish advent calendar        the scotch are nice .
scottish advent calendar        i want one
...
what 's your biggest flaw ?     i 'm a perfectionist . wait , is this a job interview ?
what 's your biggest flaw ?     i 'm too trusting of people .
...
```
The stylized corpus should also be in the format of TSV that each utterance on its own line.

### Models
GPT2-small and DialoGPT-medium are the backbone networks of our approach. You can download them using the script `download_resources.py`.
```
python download_resources.py --resource models.dialogpt-medium --output models/medium
python download_resources.py --resource models.gpt2-small --output models/small
```

## Style LM P(S)
The style language model could encourage the generation model to pick words expressing the desired style at the word level. It is trained by 
```bash
python style_lm/train_style_lm.py 
        --train_data_file /path/to/the/stylized/training/file
        --eval_data_file /path/to/the/stylized/validation/file
        --output_dir /path/to/the/output/directory 
        --logdir /path/to/the/logdir 
        --model_type gpt2 
        --line_by_line  
        --do_train 
        --per_gpu_train_batch_size 20 
        --model_name_or_path models/small 
        --fp16 
        --evaluate_during_training 
        --num_train_epochs 10
```

## Discriminator P(S|X)
The discriminator P(S|X) is trained to predict whether the input X matches the style S and guides P(Y|X) towards the direction of style S at the sentence level.

First, you can build the training data with samples from the stylized corpus and samples from the non-stylized dialogue corpus:

``` bash
python style_discriminator/build_data.py 
        --style_data /path/to/the/stylized/training/file
        --reddit_data /path/to/the/dialogue/training/file
        --ratio 5
        --res_fi /path/to/dump/the/discriminator/training/data
```

Then, the discriminator is trained by 
``` bash
python style_discriminator/train_style_dis.py
        --dataset_fp /path/to/the/discriminator/training/data 
        --model_name_or_path models/small 
        --output_dir /direcotry/to/dump/models
```
After training, you can use the script [style_discriminator/eval_style_dis.py](https://github.com/TobeyYang/StyleDGPT/blob/main/style_discriminaotr/eval_style_dis.py)
* to score input sentences interactively. 
```
python style_discriminator/eval_style_dis.py --mode interact --model_name_or_path models/small --sty_dic_model_fi /path/to/dumped/model
```

* to evaluate a TSV file.
```
python style_discriminator/eval_style_dis.py --mode eval --model_name_or_path models/small --sty_dic_model_fi /path/to/dumped/model --hyp /path/to/the/tsv/file
```

* to filter the data in a TSV file.
```
python style_discriminator/eval_style_dis.py --mode filter --model_name_or_path models/small --sty_dic_model_fi /path/to/dumped/model --src /path/to/the/src/file --res /path/to/the/result/file --threshold 0.4
```

## StyleDGPT Training
When the language model and discriminator are ready, you can train the StyleDGPT with
```
python style_dialogpt/train_gumbel_kl.py 
        --config configs/arxiv.json 
        --output /path/to/output/directory 
        --learning_rate 5e-7 
        --dis_scale 5e-2 
        --kl_scale 5e-4
        --ce_scale 1 
        --num_optim_steps 120000 
```
Notes:
* Refer to the config file for other key parameters.
* Validation and checkpoint saving happens according to `valid_step` parameter value.
* Every evaluation saves a model checkpoint.
* There is no stop condition besides a specified amount of steps to train (i.e., `num_optim_steps`).
* You can monitor the training process through the tensorboard, and we also output some samples from the validation set.
* `learning_rate`, `dis_scale`, and `kl_scale` are key parameters to the model performance that need to be carefully tuned on your datasets.


## StyleDGPT Inference
```
python style_dialogpt/evaluate.py
    --model_name_or_path models/medium
    --sty_lm_model_name_or_path /path/to/stylized/lm/dir
    --sty_dic_model_fi /path/to/dumped/discriminator
    --load_checkpoint models/medium/medium_ft.pkl
    --eval_input_file /path/to/inference/data/file
    --output_fi /path/to/output/file
    --temperature 1
    --top_k 40
    --return_num 50
```

## Citation
If you find this paper or this code useful, please cite:
```
@misc{yang2020styledgpt,
      title={StyleDGPT: Stylized Response Generation with Pre-trained Language Models}, 
      author={Ze Yang and Wei Wu and Can Xu and Xinnian Liang and Jiaqi Bai and Liran Wang and Wei Wang and Zhoujun Li},
      year={2020},
      eprint={2010.02569},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```





 
