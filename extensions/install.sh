#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install --user

# PointNet++
cd $HOME/extensions/pointnet2_ops_lib
python setup.py install --user

