

1. Install PosePipeline into your path

```
git clone https://github.com/peabody124/PosePipeline.git
cd PosePipeline
pip install -e .
```

2. Launch DataJoint database. One can also set up a local MySQL database, following the instruction at DataJoint.

```
cd datajoint_docker
docker-compose up -d
```

3. When running code, make sure to configure repository for where video stores will be kept. This can also be saved to 
   the DataJoint configuration.

```
dj.config['stores'] = {
    'localattach': {
        'protocol': 'file',
        'location': '/mnt/data0/clinical_data/datajoint_external'
    }
}
```

4. To use `OpenMMLab` packages, the following additional step is required. 

```
bash scripts/mmlab_install_mim.sh
```
