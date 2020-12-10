# Conv-EncDec-PyTorch

If you use our code or data in your work, please do cite our paper

```
@article{maqbool2020m2caiseg,
  title={m2caiSeg: Semantic Segmentation of Laparoscopic Images using Convolutional Neural Networks},
  author={Maqbool, Salman and Riaz, Aqsa and Sajid, Hasan and Hasan, Osman},
  journal={arXiv preprint arXiv:2008.10134},
  year={2020}
}
```

\
 \
 \
 \
 \
 \
 \
 \
 \
**IGNORE BELOW - IT IS OUTDATED**
 

### Deprecated readme - please ignore
[WIP]
[Detailed Readme to be added]

[ToDo]
1. Update the Readme 
  -> Provide detailed instructions on how to run and where to make changes
  -> List dependencies
2. Use a different loss than MSELoss
3. Remove unnecessary code/comments and add more helpful comments
4. Implement code to train and evaluate on part of dataset (in between epochs) and make it a parameter
5. Try more transforms
6. Make displaying images a parameter
7. Nice graphs -> Possibly use Visdom or Tensorboard

Convolutional Encoder Decoder network based on the SegNet architecture for unsupervised feature learning

This SegNet based Convolutional Encoder-Decoder network can be used for unsupervised feature learning for particular datasets.
The PyTorch Dataset class for the MICCAI dataset has been provided. You can make the changes in that to adapt to your dataset.

A sample training script is also provided.
