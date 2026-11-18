# Code understanding — likely questions and answers

## 1. Why are there two networks?
The online network is updated during replay. The target network is copied from the online network only at synchronization points. Holding the target fixed between copies makes the Q-learning target more stable.

## 2. Why transpose the input?
The environment returns `(N, H, W, C)` data, the channels-last layout used by TensorFlow/Keras. PyTorch convolution layers expect `(N, C, H, W)`, so `_prepare_input()` changes the layout before inference and training.

## 3. Why permute back before Flatten?
Keras `Flatten` reads features in H-W-C order. PyTorch convolution outputs are N-C-H-W. Because the dense-layer weights were imported from Keras, `DQN.forward()` permutes the convolution output back to N-H-W-C before flattening. Without this step, the dense weights would receive features in the wrong order.

## 4. Why divide board values by four?
The environment encodes board information with values up to four. Dividing by four normalizes the input, matching the original v17.1 implementation.

## 5. What do the four outputs represent?
They are Q-values for the four possible actions. The agent masks illegal moves with negative infinity and chooses the legal action with the highest predicted value.

## 6. Why store legal moves in replay memory?
The next-state target must maximize only over actions that are legal in the next state. The replay buffer stores those legal-action flags with each transition.

## 7. What is the DQN target?
For a non-terminal transition it is `r + gamma * max_legal_a Q_target(s', a)`. For terminal transitions, the future-value term is removed by multiplying it by `(1 - done)`.

## 8. Why use Huber loss?
Huber loss is quadratic for small errors and linear for large errors. It is less sensitive to large temporal-difference errors than ordinary squared error. `SmoothL1Loss(beta=1)` is the PyTorch equivalent used here.

## 9. Why RMSprop?
The reference v17.1 configuration uses RMSprop with learning rate `5e-4`. The PyTorch implementation uses the same learning-rate scale, with decay `alpha=0.9` and epsilon `1e-7`.

## 10. How was the reference checkpoint converted?
Keras convolution kernels were transposed from HWIO to OIHW. Dense matrices were transposed from `(in, out)` to `(out, in)`. The online and target networks were compared tensor by tensor against the original HDF5 weights; all ten tensors in each network had maximum absolute difference zero.

## 11. What was changed from the original project?
The DQN layers, inference, loss, optimizer, training update, and checkpoint handling were rewritten with PyTorch. The Snake environment and replay-buffer interface were retained as far as practical. The feature ordering before Flatten was explicitly corrected to preserve the Keras dense-weight semantics.

## 12. What do the checkpoints contain?
`model_0000.pth` contains the converted v17.1 reference online and target weights. `model_0400.pth` contains the resulting PyTorch fine-tuning checkpoint after four hundred updates. Both store model weights, optimizer state, dimensions, action count, gamma, and version metadata.
