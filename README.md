## Learning Transformers

> 项目采用 [PDM](https://pdm-project.org/) 作为包管理器

### 课程 
- [NLP Course](https://huggingface.co/learn/nlp-course/zh-CN/chapter0/1)
- [Transformers](https://huggingface.co/docs/transformers/v4.44.0/zh/index)
- [Transformers 快速入门](https://transformers.run/)

### 其它

#### PDM配置

##### 配置 pypi 镜像 

```
pdm config pypi.url	https://pypi.tuna.tsinghua.edu.cn/simple
```

OR 

```
export PDM_PYPI_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 导出 requirements.txt

```
pdm export -o requirements.txt --without-hashes
```

#### HF配置

##### HF镜像

```
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://hf-mirror.com
```