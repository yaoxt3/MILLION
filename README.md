# MILLION
An implementation of [MILLION](https://arxiv.org/abs/2209.04924)

## Citation
```
@inproceedings{bing23million,
  title={Meta-Reinforcement Learning via Language Instructions},
  author={Zhenshan Bing, Alexander Koch, Xiangtong Yao, Kai Huang and Alois Knoll},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  address = {London, UK}
}
```

## Abstract
Although deep reinforcement learning has recently been very successful at learning complex behaviors, it requires a tremendous amount of data to learn a task. One of the fundamental reasons causing this limitation lies in the nature of
the trial-and-error learning paradigm of reinforcement learning, where the agent communicates with the environment and progresses in the learning only relying on the reward signal. This is implicit and rather insufficient to learn a task well. On the contrary, humans are usually taught new skills via natural language instructions. Utilizing language instructions for robotic motion control to improve the adaptability is a recently emerged topic and challenging. In this paper, we present a meta-RL algorithm that addresses the challenge of learning skills with language instructions in multiple manipulation tasks. On the one hand,
our algorithm utilizes the language instructions to shape its interpretation of the task, on the other hand, it still learns to solve task in a trial-and-error process. We evaluate our algorithm on the robotic manipulation benchmark (Meta-World) and it significantly outperforms state-of-the-art methods in terms of training and testing task success rates. 

## Requirements
1. Linux system
2. Python 3.8.5
3. rlpyt
4. Meta-World
5. Mujoco-py

## Installation
```bash
1. conda create -n metalearn python==3.8.5

2. source activate metalearn

for GPU version:
3. conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

for CPU version:
3. conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cpuonly -c pytorch

To install rlpyt:
4. pip install atari-py pyprind opencv-python psutil

5. cd ./rlpyt

6. export PYTHONPATH=path_to_rlpyt:$PYTHONPATH
PS:the path_to_rlpyt is the path to rlpyt folder, like xx/rlpyt

7. pip install -e .

To install Mujoco-py:
8. please reference to [Mujoco-py 2.1](https://github.com/openai/mujoco-py)

To install meta-world:
9. cd ./metaWorld && pip install -e .

10. git checkout d9a75c451a15b0ba39d8b7a8b6d18d883b8655d8
PS: [Meta-World](https://github.com/rlworkgroup/metaworld) (MuJoCo dependencies required) (python 3.8.5 and torch 1.8.1 are required) 

11. cd MILLION && pip install -e .
```
In order to use the Meta-World environments with language instructions the word vector representations have to be saved first.
This is done by executing:
```
python3 MILLION/learning_to_be_taught/environments/meta_world/save_used_word_embeddings.py.
```

## Experiments


### Meta-World ML10 training
This experiment is in the experiments/ml10_demonstrations directory.
run_experiments.sh starts all the training runs serially. 

run the following command in the experiments/ml10_demonstrations directory to train the policy using language instructions：
```
$ python3 ml10_language_instructions.py --log_dir=$DIR/../logs/ml10/language_instructions
```

### Visulaization
```
$ python3 visualize_policy.py 
```

If you want to visual the interaction results, set render=True (line 148), and you could set the path of trained policy on line 95.

For More information about gym, refer to http://gym.openai.com/docs/ and https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

### Documents
Some documents are saved in the ./Documents directory. And I also recommend a series of videos to learn RL (https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLdbehimOkuLqfrq7BF0c8kHu1Y9pBWE9z&index=1&t=680s).
