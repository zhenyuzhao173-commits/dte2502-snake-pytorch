# Running Graded assignment 02

This is the DTE-2502 Neural Networks Graded Assignment 02 project. It ports the
reference Snake `DeepQLearningAgent` from TensorFlow/Keras to PyTorch and uses
the required **v17.1** CNN configuration.

## 1. Environment setup

Use the course GA01 conda environment so the project is tested with the same
Python/PyTorch environment expected by the lecturer. Activate that environment,
then install only missing packages from:

```bash
pip install -r requirements.txt
```

The final project runtime uses Python and PyTorch; it does not import
TensorFlow/Keras. The code defaults to CUDA when available and otherwise uses
CPU.

## 2. Run a quick smoke test

From this project directory:

```bash
python -m unittest discover -s tests -v
python evaluate.py --checkpoint 400 --games 30 --seed 42
```

The evaluation uses the `max_time_limit` and obstacle setting from
`model_config/v17.1.json` so that the test environment matches the configured
v17.1 environment.

## 3. Fine-tune the converted v17.1 model

The project includes the converted reference checkpoint at
`models/v17.1/model_0000.pth`. To reproduce the included 400-update PyTorch
fine-tuning run:

```bash
python training.py --version v17.1 --episodes 400 --games 16 --eval-games 30 --log-frequency 100 --seed 42 --init models/v17.1 --epsilon-start 0.01 --epsilon-decay 1.0 --epsilon-end 0.01
```

The resulting checkpoint is `models/v17.1/model_0400.pth`. Training metrics are
written to `results/training_v17.1.csv`. The included `model_0400.pth` was
created using the corrected feature ordering described in
`MODEL_CONVERSION_NOTES.md`.

For a short smoke-training run, use:

```bash
python training.py --version v17.1 --episodes 2 --games 4 --eval-games 4 --log-frequency 1 --seed 42 --init models/v17.1 --epsilon-start 0.01 --epsilon-decay 1.0 --epsilon-end 0.01
```

This smoke command writes temporary `model_0001.pth` and `model_0002.pth`
checkpoints. Do not confuse those smoke-test outputs with the submitted
`model_0400.pth`.

## 4. Evaluate a checkpoint

```bash
python evaluate.py --checkpoint 400 --games 30 --seed 42
```

The program prints the number of completed games, mean reward and mean snake
length. The recorded comparison of the converted reference checkpoint and the
fine-tuned checkpoint is in `results/evaluation_summary.csv`. Results from a
stochastic game environment can vary with the seed and evaluation batch.

## 5. Visualize the final policy

```bash
python policy_visualization.py --checkpoint 400 --steps 998 --attempts 12
```

This creates `results/final_policy.gif` and `results/final_policy.png`. The GIF
is an example rollout; it is a visual demonstration and should not be confused
with the aggregate 30-game evaluation statistics.

## 6. Code flow

1. `game_environment.py` produces the Snake state as two consecutive frames
   with shape `(N, H, W, C)`.
2. `DeepQLearningAgent` normalizes board values by four and converts the input
   to PyTorch's `(N, C, H, W)` format.
3. The CNN predicts four Q-values; illegal moves are masked before `argmax`.
4. `utils.play_game2()` collects transitions in `ReplayBufferNumpy`.
5. `train_agent()` samples a minibatch, forms the DQN target using the target
   network and minimizes Huber loss through PyTorch `SmoothL1Loss`.
6. The online network is copied to the target network at the configured
   synchronization interval.
7. Checkpoints store the online network, target network, optimizer state and
   model metadata.

## 7. CNN architecture and implementation changes

The required v17.1 topology is:

- Input: 10 x 10 board, two frames, four actions.
- `Conv2d`: 2 to 16 channels, 3 x 3 kernel, same padding.
- `Conv2d`: 16 to 32 channels, 3 x 3 kernel, valid padding.
- `Conv2d`: 32 to 64 channels, 5 x 5 kernel, valid padding.
- `Flatten`: 1024 features, followed by a 64-unit dense layer and four linear
  action-value outputs.

The implementation replaces Keras layers, inference, Huber loss, RMSprop and
HDF5 checkpoint loading/saving with PyTorch equivalents. It preserves the
reference environment and replay-buffer interface as far as practical. The
NHWC ordering is restored before flattening so imported Keras dense weights
retain their original feature semantics; see `MODEL_CONVERSION_NOTES.md`.

## 8. Additional dependencies

The project uses `numpy`, `pandas`, `matplotlib`, `tqdm` and `torch`. These are
listed in `requirements.txt`. TensorFlow/Keras is not required by the final
runtime. Use the course GA01 environment as the source of compatible package
versions rather than upgrading the environment unnecessarily.

## 9. Final GitHub page

After pushing the project to your own GitHub account, replace this line with
the actual repository URL:

`YOUR_GITHUB_REPOSITORY_URL`

## 10. Included deliverables

- `agent.py` - PyTorch `DeepQLearningAgent` and v17.1 CNN.
- `training.py` - PyTorch fine-tuning loop.
- `evaluate.py` - reproducible evaluation entry point.
- `policy_visualization.py` - final-policy GIF/PNG generator.
- `model_config/v17.1.json` - required network/environment configuration.
- `models/v17.1/model_0000.pth` - converted reference online/target weights.
- `models/v17.1/model_0400.pth` - PyTorch fine-tuned checkpoint.
- `results/` - evaluation/training results and final-policy visualizations.
- `DTE2502_Graded_Assignment02_15_Slides.pptx` - exactly 15 slides.
- `PRESENTATION_SCRIPT.md` - approximately 20 seconds of narration per slide.
- `CODE_UNDERSTANDING_QA.md` - likely code-understanding questions.
- `MODEL_CONVERSION_NOTES.md` - weight conversion and tensor layout details.
