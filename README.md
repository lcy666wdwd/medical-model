# medical-model
This project develops a 1D Convolutional Neural Network (1D-CNN) for the automatic detection of sleep apnea events from raw electrocardiogram (ECG) signals, based on the PhysioNet Apnea-ECG International Open Competition Database.

The workflow includes comprehensive data preprocessing (parsing multi-format binary files, minute-by-minute segmentation, and z-score normalization) and a four-layer convolutional feature extraction network. The model learns waveform features end-to-end, eliminating the need for manually engineered features such as RR intervals or EDR.

The model achieves a competitive accuracy of 91.69% on the test set with an AUC of 0.9747, demonstrating robust training without overfitting and excellent generalization capabilities.


1Dcnn神经网络卷积模型，包含训练脚本和模型本身，以及包含预处理的调用文件。
· 基于PhysioNet Apnea-ECG国际公开竞赛数据库，设计并实现一个一维卷积神经网络（1D-CNN）模型，用于从原始心电信号中自动检测睡眠呼吸暂停事件。
· 完成数据预处理（多格式二进制解析、分钟级切分、z-score标准化），构建四层卷积特征提取网络，端到端学习波形特征，无需人工设计RR间期、EDR等传统特征。
· 模型在测试集上达到91.69%的准确率，AUC为0.9747，训练过程无过拟合，具备良好的泛化能力。
