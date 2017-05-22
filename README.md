## Curiosity-driven Exploration by Self-supervised Prediction ##
#### In ICML 2017 [[Project Website]](http://pathak22.github.io/noreward-rl/) [[Demo Video]](http://pathak22.github.io/noreward-rl/index.html#demoVideo)

[Deepak Pathak](https://people.eecs.berkeley.edu/~pathak/), [Pulkit Agrawal](https://people.eecs.berkeley.edu/~pulkitag/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)<br/>
University of California, Berkeley<br/>

<img src="images/mario1.gif" width="300">    <img src="images/vizdoom.gif" width="351">

This is the code for our [ICML 2017 paper on curiosity-driven exploration for reinforcement learning](http://pathak22.github.io/noreward-rl/). Idea is to train agent with intrinsic curiosity-based motivation (ICM) when external rewards from environment are sparse. Surprisingly, you can use ICM even when there are no rewards available from the environment, in which case, agent learns to explore only out of curiosity: 'RL without rewards'. If you find this work useful in your research, please cite:

    @inproceedings{pathakICMl17curiosity,
        Author = {Pathak, Deepak and Agrawal, Pulkit and
                  Efros, Alexei A. and Darrell, Trevor},
        Title = {Curiosity-driven Exploration by Self-supervised Prediction},
        Booktitle = {International Conference on Machine Learning ({ICML})},
        Year = {2017}
    }

### 1) Running demo

1. Install dependencies in the virtual environment and download trained models:
  ```Shell
  git clone -b master --single-branch https://github.com/pathak22/noreward-rl.git
  cd noreward-rl/
  virtualenv curiosity
  source $PWD/curiosity/bin/activate
  pip install -r src/requirements.txt
  bash ./models/download_models.sh
  ```

2. Run demo:
  ```Shell
  cd noreward-rl/src/
  python demo.py --ckpt ../models/doom/doom_ICM
  # By default, it runs doom "dense-reward" setup. If you want to run, "sparse" and "very-sparse" reward setups, replace relevant doom files as follows:
  # mv noreward-rl/curiosity/local/lib/python2.7/site-packages/doom_py/scenarios/my_way_home.wad  noreward-rl/curiosity/local/lib/python2.7/site-packages/doom_py/scenarios/my_way_home_orig.wad
  # cp noreward-rl/models/doom/wads/my_way_home_fixed13.wad noreward-rl/curiosity/local/lib/python2.7/site-packages/doom_py/scenarios/my_way_home.wad
  ```

### 2) Training code

[To be released soon. Stay tuned !]
