# Graph Neural Network for Traffic Volume Kriging

📝
> This is the code repository for our publication [''Towards better traffic volume estimation: Jointly addressing the underdetermination and nonequilibrium problems with correlation-adaptive GNNs''](https://doi.org/10.1016/j.trc.2023.104402) that is published on Transportation Research Part C.

## Motivation
This project studies two key problems with regard to traffic volume estimation: (1) underdetermined traffic flows caused by fully undetected paths that can allow arbitrary volume values without violating the conservation law, where using local side information is insufficient to tackle, and (2) nonequilibrium traffic flows arise when traffic flows vary in density over space and time due to congestion propagation delay, which produce time-shifted volume readings to varying degrees.
![image](https://github.com/tongnie/GNN4Flow/assets/97451044/8d860e5d-8b60-46bd-9f19-72b52c648428)

## Methodology
- Model structure:
  ![image](https://github.com/tongnie/GNN4Flow/assets/97451044/55605685-ceb6-4211-bd8a-884ebac7ee20)
  This figure shows the holistic architecture of STCAGCN. It consists of two basic components: (1) spatial information aggregation block for alleviating underdetermined problem with learned speed-volume relationships. (2) time-asynchronous correlations extraction      block for coping with nonequilibrium flows
- Inductive training:
  ![image](https://github.com/tongnie/GNN4Flow/assets/97451044/2d3998da-47ef-4e21-987b-c6f2bbe29673)

## Run
- We provide the Pytorch implementation of proposed STCAGCN in 'model.py' file.
- A pretrained model is provided in the 'check_point' folder.
- A JupyterNotebook demonstration of data loading, preprocessing, model configuration, training, evaluating, and visualizing is provided in 'run_STCAGCN.ipynb'.

## Citation
Please cite our paper if this repository is helpful for your study.
> @article{nie2023towards,
  title={Towards better traffic volume estimation: Jointly addressing the underdetermination and nonequilibrium problems with correlation-adaptive GNNs},
  author={Nie, Tong and Qin, Guoyang and Wang, Yunpeng and Sun, Jian},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={157},
  pages={104402},
  year={2023},
  publisher={Elsevier}
}

## Acknowledgement
Our implementations are built on top of the IGNNK repository:
https://github.com/Kaimaoge/IGNNK.

## License
This project is released under the MIT license.
