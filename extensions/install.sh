#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/chamfer_dist
python setup.py install --user

# PointNet++
cd $HOME/pointnet2_ops_lib
python setup.py install --user

