python3.8 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools wheel
pip install -r requirements.txt
pip install "gymnasium[classic-control]"
