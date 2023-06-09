# ALO-QA

This code is implemented based on a fork of [position-bias](https://github.com/dmis-lab/position-bias).


## Requirements

```
$ conda create -n position-bias python=3.6
$ conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
$ pip install -r requirements.txt
```

Note that Pytorch has to be installed depending on the version of CUDA. 

## Dataset

Five synthetic datasets are provided.


<table >
	<tbody>
		<tr>  
			<td> <b> Dataset </td>
			<td> <b> Answer Position </td>
			<td> <b> Example </td>
		</tr>
		<tr>
			<td> SQuAD-train-1st.json </td>
			<td> First sentence </td>
			<td> 28,263 </td>
		</tr>
		<tr>
			<td> SQuAD-train-2nd.json </td>
			<td> Second sentence </td>
			<td> 20,593 </td>
		</tr>
		<tr>
			<td> SQuAD-train-3rd.json </td>
			<td> Third sentence </td>
			<td> 15,567 </td>
		</tr>
		<tr>
			<td> SQuAD-train-4th.json </td>
			<td> Fourth sentence </td>
			<td> 10,379 </td>
		</tr>
		<tr>
			<td> SQuAD-train-5th.json </td>
			<td> Fith Sentence & later </td>
			<td> 12,610 </td>
		</tr>
	</tbody>
</table>

## Train

The following example train BERT on the synthetic dataset.

```
TRAIN_FILE=dataset/squad/SQuAD-train-1st.json
OUTPUT_DIR=logs/1st_bert
make train_bert TRAIN_FILE=${TRAIN_FILE} OUTPUT_DIR=${OUTPUT_DIR}
```
Set `use_alo True` in the makefile to use our ALO approach.

### Train Bias Ensemble
The following examples train bias ensemble methods (bias product, learned-mixin) on each synthetic dataset. To select a synthetic dataset, you can choose K between [1st, 2nd, 3rd, 4th, 5th].

```
K=1st
TRAIN_FILE=dataset/squad/SQuAD-train-${K}.json
STAT_FILE=dataset/squad/${K}_stat.p
OUTPUT_DIR=logs/${K}_prod
make train_prod_bert TRAIN_FILE=${TRAIN_FILE} STAT_FILE=${STAT_FILE} OUTPUT_DIR=${OUTPUT_DIR}
```

```
K=1st
TRAIN_FILE=dataset/squad/SQuAD-train-${K}.json
STAT_FILE=dataset/squad/${K}_stat.p
OUTPUT_DIR=logs/${K}_mixin
make train_mixin_bert TRAIN_FILE=${TRAIN_FILE} STAT_FILE=${STAT_FILE} OUTPUT_DIR=${OUTPUT_DIR}
```

We also provide answer statistics of the full SQuAD dataset. After download full SQuAD data, you can train the bias ensemble method with the following example.

```
TRAIN_FILE=dataset/squad/SQuAD-v1.1-train.json
STAT_FILE=dataset/squad/train_answer_stat.p
OUTPUT_DIR=logs/full_mixin
make train_mixin_bert TRAIN_FILE=${TRAIN_FILE} STAT_FILE=${STAT_FILE} OUTPUT_DIR=${OUTPUT_DIR}
```

## Acknowledgement

Special thanks to the authors of [position-bias](https://github.com/dmis-lab/position-bias), and the datasets used in this project.
