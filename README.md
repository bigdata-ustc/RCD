# RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems

This is our implementation for the paper:

Weibo Gao, Qi Liu*, Zhenya Huang, Yu Yin, Haoyang Bi, Mu Chun Wang, Jianhui Ma, Shijin Wang, and Yu Su. RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems[C]//Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021: 501-510.

Please cite our SIGIR'21 paper if you use our codes. Thanks!

Author: Weibo Gao (http://home.ustc.edu.cn/~weibogao/)

Email: weibogao@mail.ustc.edu.cn

## Environment Settings
We use Torch and DGL as the backend. 
- Torch version:  '1.7.1'
- DGL version: '0.6.1'

## Example to run the codes.
The instruction of commands.

* **Note**: Concept dependency local map has been provided (see the instruction of dataset). The construction of concept dependency relation see subsection 5.1.2 in the paper. If you need, we would release this code.

Go to the code directory:
```
cd RCD
```

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

**Note**: In subsection 4.3 (i.e., Extendable Diagnosis Layer) of the paper, Q_{e} in original MIRT represents exercise discrimination. We use a concept-related vector instead of discrimination as an implementation in the paper. RCD can be extended to the many forms of diagnosis.

## Dataset
### junyi

log_data.json:
- Student exercising records.
- Source: https://github.com/bigdata-ustc/EduData

train_set.json
- Train file.

test_set.json
- Test file.

graph/K_Directed.txt
- Prerequisite relation from concept dependency local map.
- Each line is a prerequisite relation from concept dependency local map: precursor_concept_ID\t succeed_concept_ID.

graph/K_Undirected.txt
- Similarity relation from concept dependency local map.
- Each line is a similarity relation from concept dependency local map: concept_ID\t similar_concept_ID.

**Note**: Exercise-concept correlation local map and student-exercise interaction local map can be constructed by running build_k_e_graph.py and build_u_e_graph.py respectively.

Last Update Date: December 20, 2021
