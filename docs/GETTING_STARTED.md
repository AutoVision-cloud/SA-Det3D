## Getting Started
 


### Test and evaluate the pretrained models
* Go to tools:
```
cd OpenPCDet/tools
```

* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* For example:
```shell script
python test.py --cfg_file cfgs/kitti_models/pointpillar_fsa.yaml --batch_size 4 --ckpt ${SAVED_CKPT_PATH}/pointpillar_fsa.pth
```



### Train a model
* Train with multiple GPUs:
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}  --epochs 80
```

* For example:
```shell script
sh scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/pointpillar_fsa.yaml  --epochs 80
```



* Train with a single GPU:
```shell script
python train.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --epochs 50
```

