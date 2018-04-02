# lstm-xor

Warm-up exercise from OpenAI's [Requests for Research 2.0](https://blog.openai.com/requests-for-research-2/).

Run the following to train with default parameters.

```
$ python lstm-xor.py --train
```

And the following command to test using the latest model, which allows you to enter binary sequences for the model to predict.

```
$ python lstm-xor.py --test

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

