# v17.1 checkpoint conversion notes

## Source checkpoint

The reference archive contains the v17.1 Keras weights `model_163500.h5` and
`model_163500_target.h5`. The submitted `models/v17.1/model_0000.pth` is their
PyTorch conversion. Both the online and target networks were checked against
the source HDF5 tensors after the required kernel transpositions; all ten
weight/bias tensors in each network match with a maximum absolute difference
of `0.0`.

## Tensor mapping

- Keras convolution kernels use HWIO; PyTorch `Conv2d` uses OIHW. Kernels are
  transposed from `(H, W, in_channels, out_channels)` to
  `(out_channels, in_channels, H, W)`.
- Keras dense kernels use `(in_features, out_features)`; PyTorch `Linear`
  weights use `(out_features, in_features)`, so those matrices are transposed.
- Keras consumes channels-last tensors. PyTorch convolutions consume
  channels-first tensors. The implementation converts NHWC to NCHW before
  the convolutions, then returns to NHWC before `Flatten`. This last step is
  essential because the imported dense-layer weights were trained with the
  Keras H-W-C flatten order.

## Included checkpoints

- `model_0000.pth`: converted v17.1 reference checkpoint (iteration 163500 in
  the original repository), used as the initial policy and training base.
- `model_0400.pth`: 400 PyTorch fine-tuning updates from `model_0000.pth`,
  using the corrected NHWC-compatible flatten order.

The measured 30-game evaluation is recorded in `results/evaluation_summary.csv`.
The fine-tuned checkpoint is close to the converted reference checkpoint; the
results are not presented as proof that fine-tuning improved the policy.
