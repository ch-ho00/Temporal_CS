# Temporal Commonsense understanding
This project is based on dataset and code provided by “Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding EMNLP 2019. ([link](https://arxiv.org/abs/1909.03065))
This project aims to improve the performance on the provided MCTACO dataset with two approaches.
1. Interval prediction approach

2. Data augmentation

## Dataset
Please see text_augmentation_colab.ipynb for the text augmentation approach, and the augmented data files are stored under eval_5_fold_dataset

## Experiments (WIP)
At this point, we provide the outputs of the interval_BERT and augment_BERT. 

To run interval_BERT: 

First install required packages with: 
```bash 
pip install -r experiments/bert/requirements.txt
pip install word2number 
pip install sutime 
mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')
```

Run the following code to traing : 
```
python ./experiments/bert/run_interval_classifier.py
```


Your the following command to reproduce BERT predictions under `./bert_output`: 
```bash
sh experiments/bert/run_bert_baseline.sh
```
Evaluate the predictions with which you can further evaluate with the following command: 

```bash 
python evaluator/evaluator.py eval --test_file dataset/test_9442.tsv --prediction_file bert_output/eval_outputs.txt
```

ESIM baseline: Releasing soon after some polish

## Citation
See the following paper:

```
@inproceedings{ZKNR19,
    author = {Ben Zhou, Daniel Khashabi, Qiang Ning and Dan Roth},
    title = {“Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding },
    booktitle = {EMNLP},
    year = {2019},
}
```