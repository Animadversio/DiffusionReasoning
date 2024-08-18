#!/bin/bash

tensorboard --logdir $STORE_DIR/DL_Projects/DiT/results --port 6006 &
tensorboard --logdir $STORE_DIR/DL_Projects/SiT/results --port 6007 &
tensorboard --logdir $STORE_DIR/DL_Projects/mini_edm/exps --port 6008 &
tensorboard --logdir $STORE_DIR/DL_Projects/GPT2_raven --port 6009 &
tensorboard --logdir $STORE_DIR/DL_Projects/Mamba_raven --port 6010 &
wait