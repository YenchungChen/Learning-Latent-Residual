# Boosting Image and Video Compression via Learning Latent Residual Patterns (BMVC 2020)
[Yen-Chung Chen](https://yenchungchen.github.io/),
Keng-Jui Chang,
[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/),
[Wei-Chen Chiu](https://walonchiu.github.io/)

[British Machine Vision Conference (BMVC), 2020](https://www.bmvc2020-conference.com/)

[[Paper]](https://people.cs.nctu.edu.tw/~walon/publications/chen2020bmvc.pdf)[[Video]](https://www.bmvc2020-conference.com/conference/papers/paper_0174.html)[[Supplementary]](https://people.cs.nctu.edu.tw/~walon/publications/chen2020bmvc_supp.pdf)
### Introduction
The proposed framework aims to perform frame-by-frame video compression enhancement.
We focus on utilizing the residual information, which is the difference between a compressed video and its corresponding original/uncompressed one, and propose a fairly efficient way to transmit the residual with the compressed video in order to boost the quality of video compression.
For more details, please check out our [video](https://www.bmvc2020-conference.com/conference/papers/paper_0174.html), [paper](https://people.cs.nctu.edu.tw/~walon/publications/chen2020bmvc.pdf), and [supplementary materials](https://people.cs.nctu.edu.tw/~walon/publications/chen2020bmvc_supp.pdf).

### Installation
Download repository:
```bash
git clone https://github.com/YenchungChen/Learning-Latent-Residual.git
```
Requirements are listed in `environment.yml` file.

Create the environment from the `environment.yml` using [Anaconda](https://www.anaconda.com/), and activate it:
```bash
cd Learning-Latent-Residual
conda env create -f environment.yml
conda activate latent
```

### Training

```bash
python train.py --config_filepath configs/train.yml
```

### Testing

```bash
python eval.py --config_filepath configs/test.yml
```

### Citation
If you find the code useful for your research, please cite:
```Bibtex
@inproceedings{chen20bmvc,
 title = {Boosting Image and Video Compression via Learning Latent Residual Patterns},
 author = {Yen-Chung Chen and Keng-Jui Chang and Yi-Hsuan Tsai and Wei-Chen Chiu},
 booktitle = {British Machine Vision Conference (BMVC)},
 year = {2020}
}

@inproceedings{chen19clic,
 title = {Boosting Image and Video Compression via Learning Latent Residual Patterns},
 author = {Yen-Chung Chen and Keng-Jui Chang and Yi-Hsuan Tsai and Wei-Chen Chiu},
 booktitle = {Workshop and Challenge on Learned Image Compression (CLIC, in conjunction with CVPR)},
 year = {2019}
}
```
