import json
import numpy as np
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Songti SC'] # 或者 'SimHei', 'Microsoft YaHei' 等支持中文的字体
plt.rcParams['axes.unicode_minus'] = False

# --- print_data_format 和 load_json_file 函数保持不变 ---

def print_data_format(data, name="数据"):
    """打印数据格式和前几个样本的信息"""
    if not data:
        print(f"{name}为空或未加载")
        return

    print(f"\n{name}格式信息:")
    print(f"数据类型: {type(data)}")
    print(f"样本数量: {len(data)}")

    # 打印前3个样本的所有键
    print(f"\n{name}前3个样本的键:")
    for i, item in enumerate(data[:3]):
        if isinstance(item, dict):
            print(f"样本 {i+1} 键: {list(item.keys())}")
        else:
             print(f"样本 {i+1} 类型: {type(item)}, 内容: {item}") # 处理非字典项

    # 打印前3个样本的完整内容
    print(f"\n{name}前3个样本的完整内容:")
    for i, item in enumerate(data[:3]):
        print(f"样本 {i+1}:")
        if isinstance(item, dict):
            for key, value in item.items():
                print(f"  {key}: {value}")
        else:
            print(f"  内容: {item}") # 处理非字典项


def load_json_file(file_path):
    """加载JSON文件并返回数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保返回列表
        if isinstance(data, dict):
            # 如果字典的值是列表，并且看起来像主要数据，则返回该列表
            # (这是一种启发式方法，可能需要根据你的具体JSON结构调整)
            if len(data) == 1:
                 first_value = next(iter(data.values()))
                 if isinstance(first_value, list):
                     print(f"注意: 加载的JSON是字典，但其值是列表，将使用该列表。")
                     data = first_value

            # 如果仍然是字典，将其包装在列表中（如果需要）
            # 但通常顶层是列表或单个对象字典
            # 如果你的JSON结构是 { "key1": [...], "key2": [...] } 并且你想处理其中的一个列表
            # 你需要更具体的逻辑来选择哪个列表
            elif not isinstance(data, list):
                 print(f"注意: 加载的JSON是字典，将包装在列表中。")
                 data = [data] # 如果顶层就是一个对象，将其放入列表

        elif not isinstance(data, list):
             print(f"警告: 加载的JSON既不是字典也不是列表，类型为 {type(data)}。请检查文件格式。")
             # 根据需要决定如何处理，这里返回None或原始data
             return None # 或者 return data

        return data
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析JSON文件 {file_path} - {e}")
        return None
    except Exception as e:
        print(f"加载文件 {file_path} 时发生意外错误: {e}")
        return None

# --- calculate_metrics 函数保持不变 ---
def calculate_metrics(scores, labels, threshold):
    """
    计算分类指标 (Accuracy, Precision, Recall, F1)
    假设分数小于阈值预测为正类(标签1, AI生成文本)
    """
    predictions = [1 if score < threshold else 0 for score in scores]

    # 假设标签1是正类(AI), 0是负类(Human)
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)

    # 避免除零错误
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0

    return accuracy, precision, recall, f1, predictions


def evaluate_dataset(scores_file, original_file, best_log_x, best_threshold, den_epsilon=1e-6, dataset_name="Dataset"):
    """
    评估给定数据集的性能，并按模型类型分组分析

    参数:
        scores_file: 包含Binoculars分数的文件路径
        original_file: 原始数据文件，包含模型信息
        best_log_x: Binoculars公式中的log_x参数
        best_threshold: 分类阈值
        den_epsilon: 避免除零的小值
        dataset_name: 数据集名称
    """
    # ***** 新增：定义图片输出目录 *****
    # 注意：你的文件树显示的是 imgae (可能是拼写错误)，这里使用 image
    # 如果你想使用 imgae，请将 "image" 改为 "imgae"
    output_image_dir = "image"
    # 确保目录存在，如果不存在则创建
    os.makedirs(output_image_dir, exist_ok=True)
    # ********************************

    print(f"加载{dataset_name}分数数据从{scores_file}...")
    scores_data = load_json_file(scores_file)
    # print_data_format(scores_data, f"{dataset_name}分数文件") # 移动到加载后
    if not scores_data:
        return None, None
    print_data_format(scores_data, f"{dataset_name}分数文件") # 移到这里

    print(f"加载{dataset_name}原始数据从{original_file}...")
    original_data = load_json_file(original_file)
    # print_data_format(original_data, f"{dataset_name}原始文件") # 移动到加载后
    if not original_data:
        return None, None
    print_data_format(original_data, f"{dataset_name}原始文件") # 移到这里

    # 创建id到模型和来源的映射
    id_to_info = {}
    missing_id_count = 0
    key_id = 'id' # 假设id字段名是 'id'

    # 检查原始数据是否为列表
    if not isinstance(original_data, list):
        print(f"错误: {dataset_name}原始数据不是预期的列表格式。类型: {type(original_data)}")
        return None, None

    for item in original_data:
        if not isinstance(item, dict):
            print(f"警告: {dataset_name}原始数据中发现非字典项: {item}")
            continue # 跳过非字典项

        if key_id in item:
            id_value = item[key_id]
            id_to_info[str(id_value)] = {
                'model': item.get('model', 'unknown'),
                'source': item.get('source', 'unknown')
            }
        else:
            missing_id_count += 1
            # 如果没有id字段，尝试使用text字段作为标识 (如果存在)
            if 'text' in item:
                 text = item['text']
                 # 使用文本作为标识 (注意：这可能不是唯一标识符)
                 text_id = text
                 if text_id not in id_to_info: # 避免覆盖
                     id_to_info[text_id] = {
                         'model': item.get('model', 'unknown'),
                         'source': item.get('source', 'unknown')
                     }
                 #else:
                 #   print(f"警告: 文本前缀 '{text_id}' 在原始数据中重复，可能导致信息匹配错误。")
            # else: # 如果连text都没有，记录下来
            #     print(f"警告: 原始数据样本缺少 '{key_id}' 和 'text' 字段: {item}")

    if missing_id_count > 0:
        print(f"警告: {dataset_name}原始数据中有 {missing_id_count} 个样本缺少 '{key_id}' 字段。")
        if any('text' in item for item in original_data):
             print("已尝试使用'text'字段前缀作为备用标识。")


    # 处理每个样本并合并信息
    print(f"处理{dataset_name}数据...")
    result_data = []
    missing_score_fields = 0
    missing_label_field = 0
    valid_processed_count = 0

    # 检查分数数据是否为列表
    if not isinstance(scores_data, list):
        print(f"错误: {dataset_name}分数数据不是预期的列表格式。类型: {type(scores_data)}")
        return None, None

    for item in tqdm(scores_data, desc=f"处理{dataset_name}数据"):
        if not isinstance(item, dict):
            print(f"警告: {dataset_name}分数数据中发现非字典项: {item}")
            continue # 跳过非字典项

        if 'performer_perplexity' in item and 'cross_perplexity' in item:
            if 'label' in item:
                performer_ppl = item['performer_perplexity']
                cross_ppl = item['cross_perplexity']
                label = item['label']

                # 检查数值类型
                if not isinstance(performer_ppl, (int, float)) or not isinstance(cross_ppl, (int, float)):
                    print(f"警告: 无效的困惑度值类型在样本中: {item}. 跳过此样本。")
                    continue

                # 计算Binoculars分数
                denominator = best_log_x - cross_ppl
                if denominator > den_epsilon:
                    score = performer_ppl / denominator
                    prediction = 1 if score < best_threshold else 0

                    # 查找模型和来源信息
                    model = 'unknown'
                    source = 'unknown'
                    found_info = False

                    # 尝试通过id匹配
                    if key_id in item and str(item[key_id]) in id_to_info:
                        info = id_to_info[str(item[key_id])]
                        model = info['model']
                        source = info['source']
                        found_info = True
                    # 如果没有id，尝试使用文本匹配 (如果分数文件中有text)
                    elif 'text' in item:
                        text_id = item['text'][:50]
                        if text_id in id_to_info:
                            info = id_to_info[text_id]
                            model = info['model']
                            source = info['source']
                            found_info = True

                    # if not found_info:
                    #     print(f"警告: 无法为分数样本找到匹配的原始信息 (ID: {item.get(key_id, 'N/A')}, Text Prefix: {item.get('text', 'N/A')[:10]}...)")


                    # 保存结果
                    result_data.append({
                        'true_label': label,
                        'prediction': prediction,
                        'score': score,
                        'model': model,
                        'source': source,
                        'correct': label == prediction
                    })
                    valid_processed_count += 1
                else:
                    # print(f"警告: 分母过小或为负，无法计算分数: log_x={best_log_x}, cross_ppl={cross_ppl}. 样本: {item}")
                    pass # 可以选择记录或忽略这些样本

            else:
                missing_label_field += 1
        else:
            missing_score_fields += 1

    if missing_label_field > 0:
        print(f"警告: {dataset_name}分数数据中有 {missing_label_field} 个样本缺少 'label' 字段。")
    if missing_score_fields > 0:
        print(f"警告: {dataset_name}分数数据中有 {missing_score_fields} 个样本缺少 'performer_perplexity' 或 'cross_perplexity' 字段。")


    if not result_data:
        print(f"错误: 处理后没有有效数据")
        return None, None

    print(f"成功处理了 {valid_processed_count} 个有效样本。")

    # 转换为DataFrame便于分析
    results_df = pd.DataFrame(result_data)

    # 整体评估
    true_labels = results_df['true_label'].tolist()
    predictions = results_df['prediction'].tolist()
    scores = results_df['score'].tolist()

    # 计算整体指标
    print(f"\n--- {dataset_name}集评估结果 ---")
    accuracy, precision, recall, f1, _ = calculate_metrics(scores, true_labels, best_threshold)

    print(f"使用log_x = {best_log_x:.4f} 和阈值 T = {best_threshold:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("---------------------------------------")

    # 混淆矩阵和分类报告
    cm = confusion_matrix(true_labels, predictions)
    # 确保标签存在于预测和真实值中，否则classification_report会报错
    unique_labels = np.unique(np.concatenate((true_labels, predictions)))
    target_names = [f'类别 {label}' for label in unique_labels]
    # 如果明确知道标签是0和1，可以直接指定
    if set(unique_labels).issubset({0, 1}):
        target_names = ['人类 (0)', 'AI (1)'] # 根据你的标签含义调整

    try:
        report = classification_report(true_labels, predictions, target_names=target_names, digits=4, zero_division=0)
        print(f"分类报告 ({dataset_name}):")
        print(report)
    except ValueError as e:
        print(f"无法生成分类报告: {e}")
        print("真实标签分布:", np.bincount(true_labels) if true_labels else "无")
        print("预测标签分布:", np.bincount(predictions) if predictions else "无")


    print(f"混淆矩阵 ({dataset_name}):")
    print(cm)


    # 绘制混淆矩阵
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['人类', 'AI'] if set(unique_labels).issubset({0, 1}) else unique_labels,
                    yticklabels=['人类', 'AI'] if set(unique_labels).issubset({0, 1}) else unique_labels)
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title(f'混淆矩阵 - {dataset_name}')
        plt.tight_layout()
        # ***** 修改：使用os.path.join指定完整路径 *****
        confusion_matrix_path = os.path.join(output_image_dir, f"confusion_matrix_{dataset_name}.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"混淆矩阵已保存为 {confusion_matrix_path}")
        # *****************************************
    except Exception as e:
        print(f"创建混淆矩阵可视化时出错: {e}")

    # 按模型分组分析
    if 'model' in results_df.columns and results_df['model'].nunique() > 1:
        print("\n按模型分组分析:")

        # 先检查是否包含多个不同的模型
        model_counts = results_df['model'].value_counts()
        print(f"模型分布: {model_counts.to_dict()}")

        # 只分析AI生成文本样本（标签为1）
        ai_samples = results_df[results_df['true_label'] == 1].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        if not ai_samples.empty:
            # 确保'correct'列是数值类型，以便进行聚合
            ai_samples['correct'] = ai_samples['correct'].astype(int)

            # 使用 agg 进行聚合计算
            model_groups = ai_samples.groupby('model').agg(
                样本数量=('model', 'size'),
                正确识别=('correct', 'sum')
            ).reset_index()

            # 计算检测准确率和误检数量
            model_groups['检测准确率'] = model_groups.apply(lambda row: row['正确识别'] / row['样本数量'] if row['样本数量'] > 0 else 0, axis=1)
            model_groups['误检为人类文本'] = model_groups['样本数量'] - model_groups['正确识别']

            # 重命名列以匹配之前的输出
            model_groups = model_groups.rename(columns={'model': '模型'})

            # 过滤掉 'unknown' 或空模型名
            model_groups = model_groups[~model_groups['模型'].isin(['unknown', ''])]

            # 排序
            model_groups = model_groups.sort_values(by='检测准确率', ascending=False)

            if not model_groups.empty:
                print("\n模型检测性能比较 (仅AI样本):")
                print(model_groups.to_string(index=False))

                # 可视化不同模型的检测准确率
                try:
                    plt.figure(figsize=(12, 7)) # 增加图像大小以便容纳更多模型
                    bar_plot = sns.barplot(x='模型', y='检测准确率', data=model_groups, palette="viridis") # 使用调色板
                    plt.title(f'{dataset_name}数据集上不同模型的检测准确率 (AI样本)')
                    plt.ylim(0, 1.05) # 稍微增加上限以便显示文本
                    plt.xticks(rotation=45, ha='right') # 旋转标签并右对齐

                    # 在柱状图上方添加准确率文本
                    for i, row in enumerate(model_groups.itertuples()):
                         # 检查检测准确率是否有效
                         if pd.notna(row.检测准确率):
                             bar_plot.text(i, row.检测准确率 + 0.02, f'{row.检测准确率:.3f}', # 显示3位小数
                                          ha='center', va='bottom', fontsize=9) # 调整字体大小

                    plt.tight_layout()
                    # ***** 修改：使用os.path.join指定完整路径 *****
                    model_accuracy_path = os.path.join(output_image_dir, f"model_detection_accuracy_{dataset_name}.png")
                    plt.savefig(model_accuracy_path)
                    plt.close()
                    print(f"模型检测准确率比较图已保存为 {model_accuracy_path}")
                    # *****************************************
                except Exception as e:
                    print(f"创建模型比较可视化时出错: {e}")
            else:
                print("没有有效的模型（非unknown/空）可供比较。")
        else:
             print("数据集中没有AI样本 (标签为1)，无法进行模型分组分析。")

    else:
        print("\n数据中没有找到多个不同的模型或'model'列不存在，无法进行模型分组分析")


    # 按数据来源分组分析
    if 'source' in results_df.columns and results_df['source'].nunique() > 1:
        print("\n按数据来源分组分析:")

        source_counts = results_df['source'].value_counts()
        print(f"来源分布: {source_counts.to_dict()}")

        # 确保'correct'列是数值类型
        results_df['correct'] = results_df['correct'].astype(int)

        # 计算每个来源的准确率 (这里是整体准确率，包括人类和AI样本)
        source_groups = results_df.groupby('source').agg(
            样本数量=('source', 'size'),
            正确分类数=('correct', 'sum')
        ).reset_index()

        source_groups['准确率'] = source_groups.apply(lambda row: row['正确分类数'] / row['样本数量'] if row['样本数量'] > 0 else 0, axis=1)

        # 重命名列
        source_groups = source_groups.rename(columns={'source': '数据来源'})

        # 过滤掉 'unknown' 或空来源名
        source_groups = source_groups[~source_groups['数据来源'].isin(['unknown', ''])]

        # 排序
        source_groups = source_groups.sort_values(by='准确率', ascending=False)

        if not source_groups.empty:
            print("\n不同来源的检测准确率 (整体):")
            # 选择要显示的列
            print(source_groups[['数据来源', '样本数量', '准确率']].to_string(index=False))

            # 可视化不同来源的检测准确率
            try:
                plt.figure(figsize=(10, 6))
                bar_plot = sns.barplot(x='数据来源', y='准确率', data=source_groups, palette="magma") # 使用不同调色板
                plt.title(f'{dataset_name}数据集上不同来源的检测准确率 (整体)')
                plt.ylim(0, 1.05)
                plt.xticks(rotation=45, ha='right')

                for i, row in enumerate(source_groups.itertuples()):
                     if pd.notna(row.准确率):
                         bar_plot.text(i, row.准确率 + 0.02, f'{row.准确率:.3f}',
                                      ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                # ***** 修改：使用os.path.join指定完整路径 *****
                source_accuracy_path = os.path.join(output_image_dir, f"source_detection_accuracy_{dataset_name}.png")
                plt.savefig(source_accuracy_path)
                plt.close()
                print(f"来源检测准确率比较图已保存为 {source_accuracy_path}")
                # *****************************************
            except Exception as e:
                print(f"创建来源比较可视化时出错: {e}")
        else:
            print("没有有效的来源（非unknown/空）可供比较。")
    else:
        print("\n数据中没有找到多个不同的来源或'source'列不存在，无法进行来源分组分析")


    return true_labels, predictions


# --- 主程序部分保持不变 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估Binoculars模型并按模型和来源分组分析')
    parser.add_argument('--train_scores', type=str, required=True, help='训练集分数文件路径')
    parser.add_argument('--dev_scores', type=str, required=True, help='开发集分数文件路径')
    parser.add_argument('--train_original', type=str, required=True, help='原始训练集文件路径')
    parser.add_argument('--dev_original', type=str, required=True, help='原始开发集文件路径')
    parser.add_argument('--log_x', type=float, default=7.4146, help='Binoculars公式中的log_x参数')
    parser.add_argument('--threshold', type=float, default=0.4118, help='用于分类的阈值T')

    args = parser.parse_args()

    # 检查文件是否存在
    for file_path in [args.train_scores, args.dev_scores, args.train_original, args.dev_original]:
        if not os.path.exists(file_path):
            print(f"错误: 文件未找到 {file_path}")
            exit(1)

    # 评估训练集
    print("\n================ 开始评估训练集 ================")
    train_labels, train_predictions = evaluate_dataset(
        args.train_scores, args.train_original,
        args.log_x, args.threshold,
        dataset_name="训练集"
    )
    print("================ 训练集评估完成 ================\n")

    # 评估开发集
    print("\n================ 开始评估开发集 ================")
    dev_labels, dev_predictions = evaluate_dataset(
        args.dev_scores, args.dev_original,
        args.log_x, args.threshold,
        dataset_name="开发集"
    )
    print("================ 开发集评估完成 ================\n")

