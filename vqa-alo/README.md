# ALO-VQA

This code is implemented as a fork of [CF-VQA][1] and [RUBi][2].

ALO is a simple yet effective novel loss function with adaptive loose optimization, which seeks to make the best of both worlds for question answering: in-distribution and out-of-distribution. Its main technical contribution is to reduce the loss adaptively according to the ratio between the previous and current optimization state on mini-batch training data. This loose optimization can be used to prevent non-debiasing methods from overlearning data bias while enabling debiasing methods to maintain slight bias learning.


## Summary
* [Installation](#installation)
	* [Setup and dependencies](#setup-and-dependencies)
	* [Download datasets](#download-datasets)
* [Quick start](#quick-start)
	* [Train a model](#train-a-model)
	* [Evaluate a model](#evaluate-a-model)
* [Acknowledgment](#acknowledgment)

## Installation
###  Setup and dependencies
Install Anaconda or Miniconda distribution based on Python3+ from their downloads' site.

```bash
conda  create  --name  vqa-alo  python=3.7
source  activate  vqa-alo
pip  install  -r  requirements.txt
```

For the error that `ModuleNotFoundError: No module named 'block.external'`, please follow [this issue](https://github.com/yuleiniu/cfvqa/issues/7) for the solution.

### Download datasets
Download annotations, images and features for VQA experiments:

```bash
bash  vqa-alo/datasets/scripts/download_vqa2.sh
bash  vqa-alo/datasets/scripts/download_vqacp2.sh
bash  vqa-alo/datasets/scripts/download_gqa.sh
```

## Quick start
### Train a model
The [bootstrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) file load the options contained in a yaml file, create the corresponding experiment directory and start the training procedure. For instance, you can train our best model on VQA-CP v2 (CFVQA+SUM+SMRL) by running:

```bash
python  -m  bootstrap.run  -o  vqa-alo/options/vqacp2/smrl_cfvqa_sum.yaml
```

Then, several files are going to be created in `logs/vqacp2/smrl_cfvqa_sum/`:
- [options.yaml] (copy of options)
- [logs.txt] (history of print)
- [logs.json] (batchs and epochs statistics)
-  **[\_vq\_val\_oe.json] (statistics for the language-prior based strategy, e.g., RUBi)**
-  **[\_cfvqa\_val\_oe.json] (statistics for CF-VQA)**
- [\_q\_val\_oe.json] (statistics for language-only branch)
- [\_v\_val\_oe.json] (statistics for vision-only branch)
- [\_all\_val\_oe.json] (statistics for the ensembled branch)
- ckpt_last_engine.pth.tar (checkpoints of last epoch)
- ckpt_last_model.pth.tar
- ckpt_last_optimizer.pth.tar

Many options are available in the options directory. You can change ` exp.dir `, ` model.network.base `, ` model.criterion.name `, ` model.criterion.loose_batch_num ` and other options.

```bash
python -m bootstrap.run \
-o options/vqacp2/smrl_baseline.yaml \
--exp.dir logs/vqacp2/updn_baseline_batch5\
--model.network.base updn \
--model.criterion.name ordinary_criterion_batch \
--model.criterion.loose_batch_num 5
```

### Evaluate a model
For a model trained on VQA v2 and GQA-OOD, you can evaluate your model on the validation and test set. Thanks to `--misc.logs_name`, the logs will be written in the new `logs_predicate.txt` and `logs_predicate.json` files, instead of being appended to the `logs.txt` and `logs.json` files.

```bash
python  -m  bootstrap.run  \
-o  options/vqa2/smrl_baseline.yaml  \
--exp.resume best_eval_epoch.accuracy_top1  \
--exp.dir logs/vqa2/smrl_baseline \
--dataset.train_split  '' \
--dataset.eval_split  test  \
--dataset.proc_split '' \
--misc.logs_name  test
```

There is no test set on VQA-CP v2, so the evaluation is done on the validation set.  In this example, [boostrap/run.py](https://github.com/Cadene/bootstrap.pytorch/blob/master/bootstrap/run.py) resume the best checkpoint on the validation set and start an evaluation on the testing set instead of the validation set while skipping the training set (train_split is empty). 

```bash
python  -m  bootstrap.run  \
-o  options/vqacp2/smrl_cfvqa_sum.yaml  \
--exp.resume best_eval_epoch.accuracy_top1  \
--exp.dir logs/vqacp2/smrl_cfvqa_sum \
--dataset.train_split  '' \
--dataset.eval_split  val  \
--misc.logs_name  test
```

Foe a model trained on GQA-OOD, you should run an extra command to get the final evaluation ` --type val `  or test ` --type test ` results. 

```bash
python eval_gqaood/evaluation.py \
--predictions path/of/result/answer/json/files/ \
--ood_test --type test
```

## Acknowledgment
Special thanks to the authors of [RUBi][2], [BLOCK][3], and [bootstrap.pytorch][4], and the datasets used in this research project.

[1]: https://github.com/yuleiniu/cfvqa
[2]: https://github.com/cdancette/rubi.bootstrap.pytorch
[3]: https://github.com/Cadene/block.bootstrap.pytorch
[4]: https://github.com/Cadene/bootstrap.pytorch

