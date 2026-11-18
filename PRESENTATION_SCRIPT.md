# DTE-2502 Graded Assignment 02 — 15-slide presentation script

**Target duration:** approximately 5 minutes. Speak clearly and allow a short pause at the end of each slide. Each section is written for roughly 20 seconds at a steady presentation pace.

## Slide 1 — Title
This presentation explains my DTE-2502 Graded Assignment 02. I converted the reference Snake Deep Q-Learning agent from TensorFlow and Keras to PyTorch. I will walk through the architecture, tensor conversion, replay learning, checkpoint validation, and the final policy demonstration.

## Slide 2 — Assignment goal
The assignment asks me to reproduce the reference DeepQLearningAgent in PyTorch, using a convolutional neural network as its function approximator. The required configuration is version 17.1. I also need to keep the surrounding project changes minimal, document how to run it, and visualize the final policy.

## Slide 3 — Reference project
I started from the instructor-referenced DragonWarrior15 snake-rl project. I inspected its original agent, replay buffer, Snake environment, utilities, and version 17.1 configuration. I retained the environment and transition interfaces, while rewriting the Deep Q-Learning agent and its training operations in PyTorch.

## Slide 4 — Code flow
At each step, the Snake environment returns two board frames. The agent normalizes the values, converts the tensor to PyTorch format, and predicts four action values. Illegal moves are masked before choosing an action. The transition is then stored for a later replay update.

## Slide 5 — CNN architecture
The v17.1 network begins with two input channels. It uses convolution layers with sixteen, thirty-two, and sixty-four filters, and kernel sizes three, three, and five. The final feature map contains one thousand and twenty-four values, followed by a sixty-four-unit dense layer and four Q-value outputs.

## Slide 6 — Tensor compatibility
The environment produces channels-last data, while PyTorch convolution layers expect channels-first data. I transpose the input before the CNN. One more conversion is essential: before flattening, I restore the height-width-channel ordering used by Keras. This keeps the imported dense weights mathematically consistent.

## Slide 7 — Replay memory
The replay buffer stores the current state, selected action, reward, next state, terminal flag, and legal actions in the next state. It is implemented as a fixed-size circular NumPy buffer. During training, random minibatches of sixty-four transitions are sampled to reduce correlation between consecutive steps.

## Slide 8 — Q-learning target
For a non-terminal transition, the target equals the immediate reward plus the discounted maximum legal Q-value of the next state, estimated by the target network. For a terminal transition, the future-value term is removed. Only the Q-value corresponding to the action actually taken is replaced.

## Slide 9 — Target network
DQN uses two copies of the network. The online network is updated during every replay step. The target network stays unchanged between synchronization points, so the learning target does not move after every update. In this fine-tuning run, the target weights are copied at one-hundred-update intervals.

## Slide 10 — Loss and optimizer
The reference uses a Huber-style objective and RMSprop. In PyTorch, I use Smooth L1 loss with beta equal to one, and RMSprop with a learning rate of five times ten to the minus four. The decay and epsilon settings are also aligned with the reference configuration.

## Slide 11 — Checkpointing
The original reference contains trained online and target weights in HDF5 format. I converted both sets into PyTorch state dictionaries. The initial checkpoint preserves those weights, and the second checkpoint contains four hundred additional PyTorch fine-tuning updates. Both checkpoints include model metadata and optimizer state.

## Slide 12 — Training setup
The fine-tuning run starts from the converted v17.1 checkpoint and performs four hundred updates. It uses sixteen parallel games, a batch size of sixty-four, and epsilon zero point zero one. The seed is forty-two, and training metrics are saved to a CSV file.

## Slide 13 — Results
I evaluated both checkpoints on thirty games in the v17.1 obstacle environment with the same seed. The converted checkpoint reached mean reward eighteen point nine and mean length twenty-one point nine. The fine-tuned checkpoint remained close, at eighteen point six three and twenty-one point six three.

## Slide 14 — Policy visualization
This slide shows frames from the final policy GIF. The agent reads the current board, estimates the four action values, removes illegal moves, and selects the best remaining action. The GIF is an example rollout, while the separate evaluation table reports aggregate results across thirty games.

## Slide 15 — Conclusion
The final project provides a PyTorch DeepQLearningAgent, the v17.1 configuration, converted and fine-tuned checkpoints, evaluation evidence, and a policy visualization. The runtime does not require TensorFlow or Keras. I will submit the source through GitHub and share the model and presentation video through Google Drive.
