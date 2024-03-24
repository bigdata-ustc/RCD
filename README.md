# RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems

This repository contains the implementation for the paper titled **RCD: Relation Map Driven Cognitive Diagnosis for Intelligent Education Systems**, published at `SIGIR'2021`. [[Paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462932)][[Presentation Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3404835.3462932&file=RCD.mp4)]

Authors: [Weibo Gao](https://scholar.google.com/citations?user=k19RS74AAAAJ), [Qi Liu](http://staff.ustc.edu.cn/~qiliuql) et al.

Email: weibogao@mail.ustc.edu.cn

Announcements:
--
- ### 🔈RCD has been used in the 'Learning & Practice' business of the [HUAWEI Educational Center APP](https://appgallery.huawei.com/app/C101178177?sharePrepath=ag&locale=zh_CN&source=appshare&subsource=C101178177&shareTo=copylink&shareFrom=appmarket&shareIds=74fcdcc6737c459a87ca7140baba644a_8&callType=SHARE), since 2022. 

## Environment Settings
- Torch version: '1.7.1'
- DGL version: '0.6.1'

## Example to Run the Codes
To run the codes using the Junyi dataset:
1. Navigate to the code directory:
   ```
   cd RCD/RCD
   ```
2. Create two folders '/model' and '/result':
   ```
   mkdir model result
   ```
3. Build exercise-concept correlation local map:
   ```
   python build_k_e_graph.py
   ```
4. Build student-exercise interaction local map:
   ```
   python build_u_e_graph.py
   ```
5. Train and test RCD model:
   ```
   python main.py
   ```

**Note**: Exercise-concept correlation local map and student-exercise interaction local map can be constructed by running `build_k_e_graph.py` and `build_u_e_graph.py` respectively.

## Dataset
### Junyi
- `log_data.json`: Student exercising records. [Source](https://github.com/bigdata-ustc/EduData)
- `train_set.json`: Data file for training.
- `test_set.json`: Data file for testing.
- `graph/K_Directed.txt`: Prerequisite relation from concept dependency local map, where each line is a prerequisite relation from the concept dependency local map: precursor_concept_ID\t succeed_concept_ID.
- `graph/K_Undirected.txt`: Similarity relation from concept dependency local map, where each line is a similarity relation from concept dependency local map: concept_ID\t similar_concept_ID.

**Note**: Concept dependency local map construction details are provided in the paper. 

### ASSIST
- `log_data.json`: Student exercising records.

## Related Works
- **Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis (SIGIR'2023).** [[Paper](https://dl.acm.org/doi/10.1145/3539618.3591774)][[Code](https://github.com/bigdata-ustc/TechCD)][[Presentation Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3539618.3591774&file=SIGIR23-fp1870.mp4)]
- **Zero-1-to-3: Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students towards Three Diagnostic Objectives (AAAI'2024).** [[Paper](https://arxiv.org/abs/2312.13434)][[Code](https://github.com/bigdata-ustc/Zero-1-to-3)]

## BibTex
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

### Last Update Date: March 14, 2024
