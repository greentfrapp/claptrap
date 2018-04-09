# lstm-xor

Warm-up exercise from OpenAI's [Requests for Research 2.0](https://blog.openai.com/requests-for-research-2/).

Run the following to train with default parameters.

```
$ python lstm-xor.py --train
```

or

```
$ python lstm-xor-keras.py --train
```

The latter uses the Keras library and has much cleaner code, but the dataset used for both is the same.


And the following command to test using the latest model, which allows you to enter binary sequences for the model to predict.

```
$ python lstm-xor.py --test

No path supplied, using latest model...

```

or

```
$ python lstm-xor-keras.py --test

No path supplied, using latest model...

```

Or use the `-h` argument to check how to enter parameters.

```
$ python lstm-xor.py -h
usage: lstm-xor.py [-h] [--train] [-hn HIDDENNUM] [-sl SEQLEN] [-ds DATASIZE]
                   [-bs BATCHSIZE] [-e EPOCH] [-type {fixed,variable}]
                   [--test] [-m MODELPATH]

Implementation of LSTM for XOR problem

optional arguments:
  -h, --help            show this help message and exit
  --train
  -hn HIDDENNUM, --hiddennum HIDDENNUM
  -sl SEQLEN, --seqlen SEQLEN
  -ds DATASIZE, --datasize DATASIZE
  -bs BATCHSIZE, --batchsize BATCHSIZE
  -e EPOCH, --epoch EPOCH
  -type {fixed,variable}, --lengthtype {fixed,variable}
  --test
  -m MODELPATH, --modelpath MODELPATH
```

## Some Notes

The Requests for Research problem suggested comparing two datasets and I have added a third:

1. `-type fixed` Sequences of fixed length 50 
2. `-type variable` Sequences of variable length, uniformly sampled from 1 to 50
3. `-type fair` Sequences of variable length, uniformly sampled from 1 to 50, with equal probability of generating a sample labeled 0 or 1

Considering that the LSTM is meant to model the [XOR](https://en.wikipedia.org/wiki/XOR_gate) operation, it is simple to see that the `fixed` dataset will rarely contain any samples of label `0`. More precisely, of the approximately 1e15 unique samples for binary sequences of length 50, only 2 samples will have label `0`. 

In general, probability of label `0` occuring in sequences of length n is <img src="https://raw.githubusercontent.com/greentfrapp/claptrap/master/lstm-xor/images/eqn_1.jpg" alt="2^{1-n}" width="35" height="17">.

<img src="https://raw.githubusercontent.com/greentfrapp/claptrap/master/lstm-xor/images/label_dist.png" alt="Distribution of labels in the two datasets" width="480px" height="whatever">

*Label distributions for a random sample of the `fixed`, `variable` and `fair` datasets with maximum length 50 and size 100000*

On the other hand, the `variable` dataset is far more likely to contain samples of label `0`. Considering a uniform sampling from 1 to 50 for length, the probability of label `0` occuring is 

<img src="https://raw.githubusercontent.com/greentfrapp/claptrap/master/lstm-xor/images/eqn_2.jpg" alt="0.02\sum_{i=1}^{50}{2^{1-i}} \approx 0.04" width="162" height="53">.

Finally, the `fair` dataset forces the distribution of labels to be equal, while uniformly sampling from the different lengths. This means that there are many repeated samples in the samples labeled `0`, since there are so few variations.

Since an LSTM trained on the `fixed` dataset will virtually never see a sample with label `0`, the trained LSTM will likely only output `1` for any input. This gives good performance for a similarly distributed test set but performs poorly on a `variable` test set or any test set that has a relatively high proportion of samples labeled `0`. 

In contrast, an LSTM trained on the `variable` dataset is likely to perform better due to the higher proportion (4%) of samples being of label `0`. However, it is still likely to miss out many test sequences of label `0`. 

Finally, an LSTM trained on the `fair` dataset will perform the best on a random test set, since it is trained on an even number of `0` and `1` samples and is likely to have seen and recognize all the variations of samples labeled `0`.