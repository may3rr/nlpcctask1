# AI 生成内容检测系统

## 项目背景

本项目是针对 NLPCC 2025 共享任务1 "大型语言模型生成文本检测" 的解决方案实现。该任务旨在区分大型语言模型生成的中文文本与人工撰写的文本，为应对AI生成内容带来的挑战提供技术支持。

**任务目标**：基于双目镜（Binoculars）方法，构建一个高效、鲁棒的AI生成内容检测系统，能够在各种场景下（特别是分布外数据）准确识别AI生成的中文文本。

## 解决方案概述

本项目采用了双目镜（Binoculars）检测方法，该方法通过比较观察者模型（Observer Model）和执行者模型（Performer Model）的预测分布差异来检测文本是否由AI生成。具体来说：

1. **观察者模型**：使用基础语言模型（如Qwen2.5-7B）来计算文本的基准困惑度
2. **执行者模型**：使用指令调优版语言模型（如Qwen2.5-7B-Instruct）来计算文本的执行困惑度
3. **双目镜分数**：计算执行者困惑度与交叉困惑度的比值，作为检测特征

### 双目镜方法公式

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
- 验证集F1分数 = 0.9218

为了进一步优化分类效果，我们对双目镜分数应用更复杂的变换：

$$
B_{\text{modified}}(x) = \frac{\mathrm{PPL}_q(x)}{\log_x - H(p, q, x)}
$$

其中 log_x 是一个可调参数。通过调整 log_x 的值，我们可以优化分类性能。项目中使用 `find_XT.py` 脚本自动寻找最佳的 log_x 和分类阈值 T。

## 主要工具和脚本

#### features_extract.py

该文件是项目的核心组件，实现了 `BinocularsComputer` 类，用于计算文本的双目镜分数。主要功能如下：

- **模型加载与管理**：按需加载观察者和执行者模型，并实现内存优化
- **文本处理**：将输入文本转换为tokens并进行批处理
- **分数计算**：计算perplexity、cross-perplexity和最终的binoculars分数
- **数据批处理**：支持批处理加速计算过程
- **结果保存**：将计算结果保存到指定目录

此外，文件还实现了 `BinocularsPredictor` 类，用于基于预计算的binoculars分数进行分类预测：
- 自动校准最佳分类阈值
- 执行预测并评估性能
- 生成最终提交结果

#### find_XT.py

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

该脚本的运行结果示例：
```json
{
  "best_log_x": 7.414581809045226,
  "best_threshold": 0.41181129816517914,
  "best_f1": 0.9218297625940938
}
```

这些参数随后用于最终的预测模型，以获得最佳的分类性能。

#### evaltrain.py

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

## 使用流程

项目的完整使用流程如下：
1. **预处理数据集**：使用`convert.py`将数据集的text中存在`\n`换行符的数据给清洗掉，因为观察到AI生成的文本中都存在\n,但是test set中的AI文本并没有换行符，担心模型收到这个特征的影响而影响性能。并转换格式为`jsonl`，方便利用Qwen模型进行数值计算。
    ```bash
    python process_json.py --input data/train.json --output data/train.jsonl
    ```

2. **特征提取**：使用 `features_extract.py` 计算训练集、开发集和测试集上的双目镜分数
   ```bash
   python features_extract.py --train_file ./data/train.json --dev_file ./data/dev.json --test_file ./data/test.json --output_dir ./computed_scores
   ```

3. **参数优化**：使用 `find_XT.py` 在开发集上寻找最佳的 log_x 和阈值 T 参数
   ```bash
   python find_XT.py --dev_file dev_scores.json
   ```

4. **模型评估**：使用 `evaltrain.py` 评估在训练集和开发集上的模型性能
   ```bash
   python evaltrain.py --train_scores train_scores.json --dev_scores dev_scores.json --train_original data/train.json --dev_original data/dev.json
   ```

5. **生成预测结果**：使用优化的参数对测试集进行预测
   ```bash
    python prediction.py --test_file test_scores.json --submission_file submission.json
   ```
- 使用预先计算的最佳参数（log_x 和阈值 T）对测试集数据进行预测
- 处理边缘情况（如分母接近零）
- 将预测结果格式化为官方要求的JSON格式
- 保存提交文件

## 实验结果

使用最佳参数 log_x = 7.4146 和阈值 T = 0.4118 进行评估，取得了以下性能：

**开发集结果：**
- 准确率 (Accuracy): 0.9036
- 精确率 (Precision): 0.9076 
- 召回率 (Recall): 0.9365
- F1分数: 0.9218

**训练集结果：**
- 准确率 (Accuracy): 0.8434
- 精确率 (Precision): 0.9076
- 召回率 (Recall): 0.8809
- F1分数: 0.8941

**不同模型的检测性能：**
| 模型  | 样本数量 | 正确识别 | 检测准确率 | 误检为人类文本 |
|-------|----------|----------|------------|----------------|
| glm   | 8063     | 7842     | 0.9726     | 221            |
| qwen  | 8209     | 7983     | 0.9725     | 226            |
| gpt4o | 8028     | 5581     | 0.6952     | 2447           |

**不同来源的检测准确率：**
| 数据来源 | 样本数量 | 准确率   |
|----------|----------|----------|
| csl      | 10800    | 0.9036   |
| cnewsum  | 10800    | 0.8210   |
| asap     | 10800    | 0.8056   |

## 结果分析

1. **模型分析**：基于训练集的结果，我们发现对不同模型的检测能力差异显著：
   - glm 和 qwen 模型生成的文本检测率非常高，达到约97%
   - gpt4o 模型生成的文本检测率明显较低，仅为约70%
   - 这表明 gpt4o 生成的文本更接近人类写作特点，更难被检测算法识别

2. **领域分析**：在不同文本领域上的检测性能也存在差异：
   - 学术写作 (csl) 领域检测准确率最高，达到90%以上
   - 新闻写作 (cnewsum) 和社交媒体评论 (asap) 领域的检测准确率较低
   - 这可能是因为学术文本有更严格的格式和表达规范，而社交媒体内容更加自由随意，风格多样化

3. **模型泛化性**：开发集上的F1分数比训练集高，表明模型具有良好的泛化能力，不存在明显的过拟合现象。