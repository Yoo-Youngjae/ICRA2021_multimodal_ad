# Multimodal Anomaly Detection (ICRA 2021)
![architecture](https://github.com/Yoo-Youngjae/ICRA2021_multimodal_ad/assets/44061214/500e82f9-f0ba-44dc-a1c2-59f75bfdd0d6)

## Introduction
We presented "Multimodal Anomaly Detection based on Deep Auto-Encoder for Object Slip Perception of Mobile Manipulation Robots" in ICRA 2021.

We present an anomaly detection method that utilizes multisensory data based on a deep autoencoder model.

The proposed framework integrates heterogeneous data streams collected from various robot sensors, including RGB and depth cameras, a microphone, and a force-torque sensor.

The integrated data is used to train a deep autoencoder to construct latent representations of the multisensory data that indicate the normal status.

Anomalies can then be identified by error scores measured by the difference between the trained encoder's latent values and the latent values of reconstructed input data.

Due to internal circumstances in the laboratory, the data was not made public.

### Test environment
- Ubuntu 16.04 or above
- Python 3.6

### Running Train Code

```Shell
   (venv) python novelty_detection.py
```


### Citation

If you find Multimodal Anomaly Detection useful in your research, please consider citing:

    @inproceedings{multimodal_ad,
        Author = {Youngjae Yoo and Chung-Yeon Lee and Byeong-Tak Zhang},
        Title = {Multimodal Anomaly Detection based on Deep Auto-Encoder for Object Slip Perception of Mobile Manipulation Robots},
        booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
        Year = {2021}
    }

