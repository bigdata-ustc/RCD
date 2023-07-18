# RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems

This is our implementation for the paper of **RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems** published on `SIGIR'2021`. [[PDF](https://dl.acm.org/doi/abs/10.1145/3404835.3462932)]

Please cite this paper if you use our codes. Thanks!

```
@inproceedings{gao2021rcd,
  title={RCD: Relation map driven cognitive diagnosis for intelligent education systems},
  author={Gao, Weibo and Liu, Qi and Huang, Zhenya and Yin, Yu and Bi, Haoyang and Wang, Mu-Chun and Ma, Jianhui and Wang, Shijin and Su, Yu},
  booktitle={Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval},
  pages={501--510},
  year={2021}
}
```

Author: [Weibo Gao](https://scholar.google.com/citations?user=k19RS74AAAAJ&hl=zh-CN)

Email: weibogao@mail.ustc.edu.cn

## Environment Settings
We use Torch and DGL as the backend. 
- Torch version:  '1.7.1'
- DGL version: '0.6.1'

## Example to run the codes
The instruction of commands and take Junyi dataset as an example (We will provide ASSIST dataset as soon as possible).

[//]: # (* **Note**: Concept dependency local map has been provided &#40;see the instruction of dataset&#41;. The construction of concept dependency relation see subsection 5.1.2 in the paper. If you need, we would release this code.)

Go to the code directory:
```
cd RCD/RCD
```
Create two folders '/model' and '/result'.

Build exercise-concept correlation local map:
```
python build_k_e_graph.py
```

Build student-exercise interaction local map:
```
python build_u_e_graph.py
```
Train and test RCD model:
```
python main.py
```

**Note**: In subsection 4.3 (i.e., Extendable Diagnosis Layer) of the paper, Q_{e} in original MIRT represents exercise discrimination. We use a concept-related vector instead of discrimination as an implementation in the paper. RCD can be extended to the many forms of cognitive diagnosis.

## Dataset(RCD/data)
### junyi

log_data.json:
- Student exercising records.
- Source: https://github.com/bigdata-ustc/EduData

train_set.json
- Data file for training.

test_set.json
- Data file for testing.

graph/K_Directed.txt
- Prerequisite relation from concept dependency local map.
- Each line is a prerequisite relation from concept dependency local map: precursor_concept_ID\t succeed_concept_ID.

graph/K_Undirected.txt
- Similarity relation from concept dependency local map.
- Each line is a similarity relation from concept dependency local map: concept_ID\t similar_concept_ID.

**Note**: Exercise-concept correlation local map and student-exercise interaction local map can be constructed by running build_k_e_graph.py and build_u_e_graph.py respectively.

### ASSIST


Last Update Date: March 23, 2022
