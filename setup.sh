apt-get install -y ttf-mscorefonts-installer 
apt-get install fonts-dejavu
apt-get install fonts-liberation

apt-get update
apt-get install -y poppler-utils

apt install -y libgdiplus
ln -s /usr/lib/libgdiplus.so /usr/lib/libgdiplus

pip install .[xls,ppt]
