## 1.Introduction
2016,Youtube的深度推荐系统论文《Deep Neural Networks for YouTube Recommendations》,有很多宝贵的工程细节

## 2.The Model Architecture
Youtube作为全球最大的UGC的视频网站，视频库非常之大，整个推荐过程分为两步走：
- Candidate Generation Model: 初筛，候选视频集合由millions降至hundreds，其主体是多层fc+ReLU，training时采用softmax，serving时采用nearest neighbor search.
- Ranking Model: 精排，候选视频降至dozens（推荐列表）

## 3.Engineering Details (Q & A)

## 4.Code


