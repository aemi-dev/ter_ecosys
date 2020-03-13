sudo apt-get update && sudo apt-get upgrade;
sudo apt-get install python python3 ipython ipython3;
sudo apt-get install python3-pip python-pip jupyter;

pip install --user scikit-learn numpy pandas ipywidgets pathlib cython networkx matplotlib pydotplus pydot tatsu prince colour bitarray sympy bqplot psutil lxml;
pip3 install --user scikit-learn numpy pandas ipywidgets pathlib cython networkx matplotlib pydotplus pydot tatsu prince colour bitarray sympy bqplot psutil lxml;
pip install --user --upgrade scikit-learn numpy pandas ipywidgets pathlib cython networkx matplotlib pydotplus pydot tatsu prince colour bitarray sympy bqplot psutil lxml;
pip3 install --user --upgrade scikit-learn numpy pandas ipywidgets pathlib cython networkx matplotlib pydotplus pydot tatsu prince colour bitarray sympy bqplot psutil lxml;

make run;