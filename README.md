## ColKGC

## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

## How to Run

Step 1 处理数据
```shell
python dataloader.py
```

Step 2 生成描述
```
python GlmApi.py
```

Step 3 训练文本模型
```
python main.py
```

Step 4 处理结果

```
python re'l.py
```

Step 5 训练文本模型

```
python llm_rerank.py
```

Step 6 训练文本模型

```
python llm_rerank.py
```

Step 7 最终测试

```
python evl_l.py
```

