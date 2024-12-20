# Python-level extension
cp -f $PATH_MAIN/customized_sgd.py /usr/local/lib/python3.10/dist-packages/torch/optim/sgd.py
cp -f $PATH_MAIN/customized_sparse.py /usr/local/lib/python3.10/dist-packages/torch/nn/modules/sparse.py

# C-level extension
cd $PATH_MAIN/custom-extension
python setup.py install

cd ..

# Install required package
pip install opt_einsum==3.3.0