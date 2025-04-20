
# AI 生成内容检测系统

## 1. 项目背景

本项目是针对 NLPCC 2025 共享任务1 "大型语言模型生成文本检测" 的解决方案实现。该任务旨在区分大型语言模型生成的中文文本与人工撰写的文本，为应对AI生成内容带来的挑战提供技术支持。

**任务目标**：构建一个高效、鲁棒的AI生成内容检测系统，能够在各种场景下（特别是分布外数据）准确识别AI生成的中文文本。

## 2. 解决方案概述

本项目采用了双目镜（Binoculars）检测方法，该方法通过比较观察者模型（Observer Model）和执行者模型（Performer Model）的预测分布差异来检测文本是否由AI生成。具体来说：

1. **观察者模型**：使用基础语言模型（如Qwen2.5-7B）来计算文本的基准困惑度
2. **执行者模型**：使用指令调优版语言模型（如Qwen2.5-7B-Instruct）来计算文本的执行困惑度
3. **双目镜分数**：计算执行者困惑度与交叉困惑度的比值，作为检测特征

**双目镜方法的灵感来源于:** 
[Hans et al. (2024)](https://arxiv.org/pdf/2401.12070)

### 2.1 双目镜方法公式

双目镜方法的核心公式如下：

$$
B(x) = \frac{\mathrm{PPL}_q(x)}{H(p, q, x)}
$$

其中：
- $B(x)$ 是文本 $x$ 的双目镜分数
- $\mathrm{PPL}_q(x)$ 是执行者模型 $q$ 对文本 $x$ 的困惑度 (perplexity)
- $H(p, q, x)$ 是观察者模型 $p$ 和执行者模型 $q$ 之间关于文本 $x$ 的交叉困惑度 (cross-perplexity)

困惑度计算公式：
$$
\mathrm{PPL}(x) = \exp\left(-\frac{1}{|x|} \sum \log P(x_i|x_{<i})\right)
$$

交叉困惑度计算公式：
$$
H(p, q, x) = \exp\left(-\frac{1}{|x|} \sum \sum p(j|x_{<i}) \log q(j|x_{<i})\right)
$$

对于AI生成的文本，$B(x)$ 值通常较小；而对于人类撰写的文本，$B(x)$ 值通常较大。基于这个观察，我们通过优化的公式和最佳阈值进行分类：

$$
\text{prediction} =
\begin{cases}
1\ (\text{AI生成}), & \text{if } \frac{\mathrm{PPL}_q(x)}{\log_x - H(p, q, x)} < T \\
0\ (\text{人类撰写}), & \text{if } \frac{\mathrm{PPL}_q(x)}{\log_x - H(p, q, x)} \geq T
\end{cases}
$$

最佳参数 $\log_x$ 和阈值 $T$ 通过在验证集上优化F1分数确定。

- log_x = 7.4146
- T = 0.4118
- 验证集F1分数 = 0.8980🎉🎉

为了进一步优化分类效果，我们对双目镜分数应用更复杂的变换：

$$
B_{\text{modified}}(x) = \frac{\mathrm{PPL}_q(x)}{\log_x - H(p, q, x)}
$$

其中 `log_x` 是一个可调参数，用于调整交叉困惑度的尺度，使得双目镜分数对噪声更加鲁棒。通过调整 `log_x` 的值，我们可以优化分类性能。项目中使用 `find_XT.py` 脚本自动寻找最佳的 `log_x` 和分类阈值 `T`。为了验证 `log_x` 参数的有效性，我们进行了消融实验，只搜索阈值 `T`，而保持原始的双目镜分数公式不变。实验结果表明，引入 `log_x` 参数并进行优化，可以显著提高 AI 生成文本检测系统的性能。在验证集上，原始方法（搜索 `log_x` 和 `T`）的 F1 分数为 0.8980，而只搜索 `T` 的方法的 F1 分数为 0.8483。


## 3. 主要工具和脚本

#### 3.1 features_extract.py

该文件是项目的核心组件，实现了 `BinocularsComputer` 类，用于计算文本的双目镜分数。 主要功能如下：

*   **模型加载与管理**：按需加载观察者和执行者模型，并实现内存优化。 **注意：首次运行时，此过程会使用 `modelscope` 库自动下载 Qwen2.5-7B 和 Qwen2.5-7B-Instruct 模型，预计占用磁盘空间约 30GB。**
*   **文本处理**：将输入文本转换为 tokens 并进行批处理。
*   **分数计算**：计算 perplexity、cross-perplexity 和最终的 binoculars 分数。
*   **数据批处理**：支持批处理加速计算过程。
*   **结果保存**：将计算结果保存到指定目录。

此外，文件还实现了 `BinocularsPredictor` 类，用于基于预计算的 binoculars 分数进行分类预测：

*   自动校准最佳分类阈值。
*   执行预测并评估性能。
*   生成最终提交结果。


#### 3.2 find_XT.py

该文件用于寻找最优的参数 log_x 和阈值 T，实现了一个参数搜索算法，通过优化以下改进版的双目镜公式来提高分类性能：

$$
B_{\text{modified}}(x) = \frac{\mathrm{PPL}_q(x)}{\log_x - H(p, q, x)}
$$

脚本主要功能：
- 从验证集加载预计算的双目镜分数
- 在给定范围内搜索最佳的 log_x 值
- 对每个 log_x 值，寻找最优的分类阈值 T
- 根据 F1 分数选择最佳的参数组合
- 将最佳参数保存到 JSON 文件中

该脚本的运行结果：
```json
{
  "best_log_x": 7.414581809045226,
  "best_threshold": 0.41181129816517914,
  "best_f1": 0.9218297625940938
}
```

这些参数随后用于最终的预测模型，以获得最佳的分类性能。

#### 3.3 evaltrain.py

该文件用于评估训练集和开发集上的模型性能，并进行详细的错误分析。主要功能包括：

- 使用预先计算的最佳参数（log_x 和阈值 T）进行分类预测
- 计算并展示整体分类指标（准确率、精确率、召回率、F1分数）
- 生成混淆矩阵并保存为可视化图像
- 按模型类型分组分析检测性能，发现哪些AI模型更容易被检测
- 按数据来源分组分析，评估在不同文本领域的检测效果
- 生成详细的模型性能比较图表（保存为PNG格式）

脚本输出多种可视化结果，包括：
- 混淆矩阵（`confusion_matrix_*.png`）
- 不同模型的检测准确率比较图（`model_detection_accuracy_*.png`）
- 不同数据来源的检测准确率比较图（`source_detection_accuracy_*.png`）

这些分析结果对于理解模型的优势和局限性，以及指导进一步优化非常有价值。

#### 3.4 prediction.py

该文件用于使用训练好的模型对测试集进行预测，并生成最终的提交文件。主要功能包括：

- 加载测试集数据和预计算的双目镜分数。
- 使用预先设定的最佳参数（`log_x = 7.4146` 和阈值 `T = 0.4118`）进行分类预测。
- 对双目镜分数应用变换公式。
- 处理分母接近零的边缘情况，对该样本默认预测为人类撰写，避免出现 `NaN` 值。
- 将预测结果格式化为包含 `id`、`text` 和 `label` 字段的 JSON 格式。
- 将预测结果保存为提交文件 `submission.json`。

## 4. 使用流程

项目的完整使用流程如下：

1. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

2. **预处理数据集**：使用`convert.py`将数据集的text中存在`\n`换行符的数据给清洗掉，因为观察到AI生成的文本中都存在\n,但是test set中的AI文本并没有换行符，担心模型收到这个特征的影响而影响性能。并转换格式为`jsonl`，方便利用Qwen模型进行数值计算。
    ```bash
    python src/convert.py --input data/train.json --output data/train.jsonl
    python src/convert.py --input data/dev.json --output data/dev.jsonl
    python src/convert.py --input data/test.json --output data/test.jsonl
    ```

3. **特征提取**：使用 `features_extract.py` 计算训练集、开发集和测试集上的双目镜分数
   ```bash
    python src/features_extract.py --train_file data/train.jsonl --dev_file data/dev.jsonl --test_file data/test.jsonl --output_dir features
   ```

4. **参数优化**：使用 `find_XT.py` 在开发集上寻找最佳的 log_x 和阈值 T 参数
   ```bash
    python src/find_XT.py --dev_file features/dev_scores.json --output_params_file best_binoculars_params_optimized_x.json
   ```

5. **模型评估**：使用 `evaltrain.py` 评估在训练集和开发集上的模型性能
   ```bash
    python src/evaltrain.py --train_scores features/train_scores.json --dev_scores features/dev_scores.json --train_original data/train.json --dev_original data/dev.json
   ```

6. **生成预测结果**：使用优化的参数对测试集进行预测
   ```bash
    python src/prediction.py --test_file features/test_scores.json --submission_file submission.json
   ```
   - 使用预先计算的最佳参数（log_x 和阈值 T）对测试集数据进行预测
   - 处理边缘情况（如分母接近零）
   - 将预测结果格式化为官方要求的JSON格式
   - 保存提交文件

## 5. 实验结果

使用最佳参数 log_x = 7.4146 和阈值 T = 0.4118 进行评估，取得了以下性能：

5.1 **开发集结果：**
- 准确率 (Accuracy): 0.9036
- 精确率 (Precision): 0.9022 
- 召回率 (Recall): 0.8946
- macro F1分数: 0.8980

5.2 **训练集结果：**
- 准确率 (Accuracy): 0.8434
- 精确率 (Precision): 0.9076
- 召回率 (Recall): 0.8809
- macro F1分数: 0.7971

5.3  **在`train.json`上评估不同模型和不同来源的分类准确率：**

 - **不同模型的检测性能：**
   | 模型  | 样本数量 | 正确识别 | 检测准确率 | 误检为人类文本 |
   |-------|----------|----------|------------|----------------|
   | glm   | 8063     | 7842     | 0.9726     | 221            |
   | qwen  | 8209     | 7983     | 0.9725     | 226            |
   | gpt4o | 8028     | 5581     | 0.6952     | 2447           |

 - **不同来源的检测准确率：**
   | 数据来源 | 样本数量 | 准确率   |
   |----------|----------|----------|
   | csl      | 10800    | 0.9036   |
   | cnewsum  | 10800    | 0.8210   |
   | asap     | 10800    | 0.8056   |

## 硬件和软件环境

*   **GPU:** A100-SXM4-80GB (80GB) \* 1
*   **CPU:** 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz
*   **内存:** 120GB
*   **CUDA:** 11.8
*   **Python:** 3.10 (Ubuntu 22.04)
*   **PyTorch:** 2.1.2
*   **依赖:** 详见 `requirements.txt`

## 结果分析

1. **模型分析**：基于训练集的结果，我们发现对不同模型的检测能力差异显著：
   - glm 和 qwen 模型生成的文本检测率非常高，达到约97%
   - gpt4o 模型生成的文本检测率明显较低，仅为约70%
   - 这表明 gpt4o 生成的文本更接近人类写作特点，更难被检测算法识别
   - 模型使用Qwen2.5-7B (Bai et al., 2024)预训练模型和Qwen2.5-7B- Instruct (Qwen Team, 2024)指令微调模型分别作为观测者和执行者，结果发现模型对于同属于中文模型的glm和qwen模型的分辨效果最佳。

2. **领域分析**：在不同文本领域上的检测性能也存在差异：
   - 学术写作 (csl) 领域检测准确率最高，达到90%以上
   - 新闻写作 (cnewsum) 和社交媒体评论 (asap) 领域的检测准确率较低
   - 这可能是因为学术文本有更严格的格式和表达规范，而社交媒体内容更加自由随意，风格多样化


## 引用与致谢

*   双目镜方法的灵感来源于:
    Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., ... & Goldstein, T. (2024, July). Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text. In *International Conference on Machine Learning* (pp. 17519-17537). PMLR.  [论文链接](https://arxiv.org/pdf/2401.12070)

*   使用了 Qwen2.5-7B 和 Qwen2.5-7B-Instruct 模型 (Qwen Team, 2024; Bai et al., 2024):

    *   Qwen Team. (2024, September). Qwen2.5: A Party of Foundation Models. [项目地址](https://qwenlm.github.io/blog/qwen2.5/)

    *   Bai, A. Y., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C., ... & Fan, Z. (2024). Qwen2 Technical Report. *arXiv preprint arXiv:2407.10671*.[论文链接](https://arxiv.org/pdf/2412.15115)

*   Wu, J., Yang, S., Zhan, R., Yuan, Y., Chao, L. S., & Wong, D. F. (2025). A survey on LLM-generated text detection: Necessity, methods, and future directions. Computational Linguistics, 1-66.[论文链接](https://direct.mit.edu/coli/article/51/1/275/127462/A-Survey-on-LLM-Generated-Text-Detection-Necessity)
*   Wu, J., Zhan, R., Wong, D. F., Yang, S., Yang, X., Yuan, Y., & Chao, L. S. (2024). DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios. In The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.[论文链接](https://arxiv.org/pdf/2410.23746)

特别感谢 NLP2CT Lab, University of Macau 组织了本次共享任务，并感谢 Derek, Fai Wong, Junchao Wu, Runzhe Zhan, Yulin Yuan 提供的支持。如有任何问题，请联系 nlp2ct.junchao@gmail.com。
