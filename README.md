 <h1 align="center"> Project SSHAD </h1>
This is an official implementation of SSAHD in our TGRS 2025 paper " Adaptive dual-domain learning for hyperspectral
 anomaly detection with state space models ".



Abstract

Recently, learning-based hyperspectral anomaly de tection (HAD) methods have demonstrated outstanding per formance, dominating mainstream research. However, existing learning-based approaches still have two issues: 1. They rarely consider both the spatial sparsity and inter-spectral similarity of hyperspectral imagery (HSI) simultaneously; 2. They treat all re gions equally, often overlooking the importance of high-frequency information in HSI, which is key to distinguish background and anomalies. To address these challenges, we propose a novel HAD method based on spatial-spectral adaptive dual-domain learning, termed SSHAD. Specifically, we first introduce the spatial-wise Selected State Space Module (SSSM) with linear complexity and the spectral-wise Frequency Division Self-Attention Module (FDSM), which are combined in parallel to form a Spatial-Spectral Block (SS-Block). The SSSM captures the global receptive field by scanning the HSI spatial dimension through a multi-directional scanning mechanism. The FDSM extracts high-frequency and low-frequency information from the HSI via the discrete wavelet transform and applies multi-scale convolution and self-similarity attention respectively, ensuring the suppression of anomalies during the reconstruction process. This parallel structure enables the network to model cross-window connections, expanding its receptive field while maintaining linear complexity. We use the SS-Block as the main component of our adaptive dual-domain learning network, forming SSHAD. Furthermore, we introduce a frequency-wise loss function to inhibit the reconstruction of high-frequency anomalies during background reconstruction. Comprehensive experiments conducted on four public datasets and two unmanned aerial vehicle-borne datasets validate the superiority and effectiveness of SSHAD. 

## Citation and Contact

If you use our work, please also cite the paper:

```
@article{liu2025adaptive,
  title={Adaptive dual-domain learning for hyperspectral anomaly detection with state space models},
  author={Liu, Sitian and Peng, Lintao and Chang, Xuyang and Wen, Guanghui and Zhu, Chunli},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  doi={10.1109/TGRS.2025.3530397},
  publisher={IEEE}
}

```


## Requirements

Conda and Pytorch. 



## Running experiments


```
 cd <path>
 # activate your virtual environment
 conda activate your_env_name
 # run experiment
 python SSHAD_train.py
```


The UHAD-U-I and UHAD-U-II datasets in our paper can be downloaded in the './data'.



## Acknowledgement

The authors would like to thank the authors of “Auto-AD” and “Deep hyperspectral prior: denoising, inpainting, super-resolution” for sharing their codes. 
