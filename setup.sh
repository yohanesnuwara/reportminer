# Run this shell file inside reportminer folder 
apt-get install -y ttf-mscorefonts-installer 
apt-get install fonts-dejavu
apt-get install fonts-liberation

apt-get update
apt-get install -y poppler-utils

apt install -y libgdiplus
ln -s /usr/lib/libgdiplus.so /usr/lib/libgdiplus

python3 -m venv .venv
. .venv/bin/activate
pip install Flask
pip install matplotlib
pip install .[all]
