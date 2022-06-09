# install blender
#wget https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz
#tar -xf blender-3.0.1-linux-x64.tar.xz 

# git clone dependencies
sudo pip install vcstool 
vcs import . < blender_dependencies.repos

# pip install dependencies
./blender-3.0.1-linux-x64/3.0/python/bin/python3.9 -m ensurepip

blender-3.0.1-linux-x64/3.0/python/bin/pip3 install blenderproc
./blender-3.0.1-linux-x64/3.0/python/bin/pip3 install ./airo_blender_toolkit/
./blender-3.0.1-linux-x64/3.0/python/bin/pip3 install -e ./towel # don't forget the -e!...

# get the textures and HDRIs
python utils/fetch_blender_polyhaven_assets.py

# get the distractor objects
mkdir assets/thingi10
python utils/Thingi10K_genus_lt_3.py --output ./assets/thingi10

echo "installation finished"