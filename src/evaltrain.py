import json
import numpy as np
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

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
        print(f"样本 {i+1} 键: {list(item.keys())}")
    
    # 打印前3个样本的完整内容
    print(f"\n{name}前3个样本的完整内容:")
    for i, item in enumerate(data[:3]):
        print(f"样本 {i+1}:")
        for key, value in item.items():
            print(f"  {key}: {value}")


def load_json_file(file_path):
    """加载JSON文件并返回数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保返回列表
        if isinstance(data, dict):
            data = [data]
        return data
    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {file_path}")
        return None

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
    print(f"加载{dataset_name}分数数据从{scores_file}...")
    scores_data = load_json_file(scores_file)
    if not scores_data:
        return None, None
    
    print(f"加载{dataset_name}原始数据从{original_file}...")
    original_data = load_json_file(original_file)
    # 在load_json_file函数之后立即添加这段代码
    scores_data = load_json_file(scores_file)
    print_data_format(scores_data, f"{dataset_name}分数文件")

    original_data = load_json_file(original_file)
    print_data_format(original_data, f"{dataset_name}原始文件")
    if not original_data:
        return None, None
    
    # 创建id到模型和来源的映射
    id_to_info = {}
    for item in original_data:
        if 'id' in item:
            id_value = item['id']
            id_to_info[str(id_value)] = {
                'model': item.get('model', 'unknown'),
                'source': item.get('source', 'unknown')
            }
    
    # 如果原始数据没有id字段，尝试使用text字段作为标识
    if not id_to_info:
        for item in original_data:
            if 'text' in item:
                text = item['text']
                # 使用文本前50个字符作为标识
                text_id = text[:50]
                id_to_info[text_id] = {
                    'model': item.get('model', 'unknown'),
                    'source': item.get('source', 'unknown')
                }
    
    # 处理每个样本并合并信息
    print(f"处理{dataset_name}数据...")
    result_data = []
    
    for item in tqdm(scores_data, desc=f"处理{dataset_name}数据"):
        if 'performer_perplexity' in item and 'cross_perplexity' in item and 'label' in item:
            performer_ppl = item['performer_perplexity']
            cross_ppl = item['cross_perplexity']
            label = item['label']
            
            # 计算Binoculars分数
            denominator = best_log_x - cross_ppl
            if denominator > den_epsilon:
                score = performer_ppl / denominator
                prediction = 1 if score < best_threshold else 0
                
                # 查找模型和来源信息
                model = 'unknown'
                source = 'unknown'
                
                # 尝试通过id匹配
                if 'id' in item and str(item['id']) in id_to_info:
                    info = id_to_info[str(item['id'])]
                    model = info['model']
                    source = info['source']
                # 如果没有id，尝试使用文本匹配
                elif 'text' in item:
                    text_id = item['text'][:50]
                    if text_id in id_to_info:
                        info = id_to_info[text_id]
                        model = info['model']
                        source = info['source']
                
                # 保存结果
                result_data.append({
                    'true_label': label,
                    'prediction': prediction,
                    'score': score,
                    'model': model,
                    'source': source,
                    'correct': label == prediction
                })
    
    if not result_data:
        print(f"错误: 处理后没有有效数据")
        return None, None
    
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
    report = classification_report(true_labels, predictions, digits=4)

    print(f"混淆矩阵 ({dataset_name}):")
    print(cm)
    print(f"分类报告 ({dataset_name}):")
    print(report)

    # 绘制混淆矩阵
    try:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['人类', 'AI'], yticklabels=['人类', 'AI'])
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.title(f'混淆矩阵 - {dataset_name}')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{dataset_name}.png")
        plt.close()
        print(f"混淆矩阵已保存为 confusion_matrix_{dataset_name}.png")
    except Exception as e:
        print(f"创建混淆矩阵可视化时出错: {e}")

    # 按模型分组分析
    if 'model' in results_df.columns and results_df['model'].nunique() > 1:
        print("\n按模型分组分析:")
        
        # 先检查是否包含多个不同的模型
        model_counts = results_df['model'].value_counts()
        print(f"模型分布: {model_counts.to_dict()}")
        
        # 只分析AI生成文本样本（标签为1）
        ai_samples = results_df[results_df['true_label'] == 1]
        model_groups = ai_samples.groupby('model')
        
        model_comparison = []
        
        for model_name, group in model_groups:
            # 跳过unknown或者空模型
            if model_name in ['unknown', '']:
                continue
                
            model_correct = group['correct'].sum()
            model_accuracy = model_correct / len(group) if len(group) > 0 else 0
            
            model_comparison.append({
                '模型': model_name,
                '样本数量': len(group),
                '正确识别': model_correct,
                '检测准确率': model_accuracy,
                '误检为人类文本': len(group) - model_correct
            })
            
            # 输出详细信息
            print(f"\n模型: {model_name}")
            print(f"AI样本数量: {len(group)}")
            print(f"正确识别为AI: {model_correct}")
            print(f"检测率: {model_accuracy:.4f}")
        
        # 将模型比较数据转换为DataFrame并排序
        if model_comparison:
            comparison_df = pd.DataFrame(model_comparison)
            comparison_df = comparison_df.sort_values(by='检测准确率', ascending=False)
            print("\n模型检测性能比较:")
            print(comparison_df.to_string(index=False))
            
            # 可视化不同模型的检测准确率
            try:
                plt.figure(figsize=(10, 6))
                bar_plot = sns.barplot(x='模型', y='检测准确率', data=comparison_df)
                plt.title(f'{dataset_name}数据集上不同模型的检测准确率')
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                
                # 在柱状图上方添加准确率文本
                for i, row in enumerate(comparison_df.itertuples()):
                    bar_plot.text(i, row.检测准确率 + 0.02, f'{row.检测准确率:.4f}', 
                                 ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"model_detection_accuracy_{dataset_name}.png")
                plt.close()
                print(f"模型检测准确率比较图已保存为 model_detection_accuracy_{dataset_name}.png")
            except Exception as e:
                print(f"创建模型比较可视化时出错: {e}")
    else:
        print("\n数据中没有找到多个不同的模型，无法进行模型分组分析")
                
    # 按数据来源分组分析
    if 'source' in results_df.columns and results_df['source'].nunique() > 1:
        print("\n按数据来源分组分析:")
        
        source_counts = results_df['source'].value_counts()
        print(f"来源分布: {source_counts.to_dict()}")
        
        # 计算每个来源的准确率
        source_groups = results_df.groupby('source')
        
        source_comparison = []
        
        for source_name, group in source_groups:
            # 跳过unknown或者空来源
            if source_name in ['unknown', '']:
                continue
                
            source_accuracy = group['correct'].sum() / len(group) if len(group) > 0 else 0
            
            source_comparison.append({
                '数据来源': source_name,
                '样本数量': len(group),
                '准确率': source_accuracy
            })
        
        # 将来源比较数据转换为DataFrame并排序
        if source_comparison:
            source_df = pd.DataFrame(source_comparison)
            source_df = source_df.sort_values(by='准确率', ascending=False)
            print("\n不同来源的检测准确率:")
            print(source_df.to_string(index=False))
            
            # 可视化不同来源的检测准确率
            try:
                plt.figure(figsize=(10, 6))
                bar_plot = sns.barplot(x='数据来源', y='准确率', data=source_df)
                plt.title(f'{dataset_name}数据集上不同来源的检测准确率')
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                
                for i, row in enumerate(source_df.itertuples()):
                    bar_plot.text(i, row.准确率 + 0.02, f'{row.准确率:.4f}', 
                                 ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f"source_detection_accuracy_{dataset_name}.png")
                plt.close()
                print(f"来源检测准确率比较图已保存为 source_detection_accuracy_{dataset_name}.png")
            except Exception as e:
                print(f"创建来源比较可视化时出错: {e}")
    else:
        print("\n数据中没有找到多个不同的来源，无法进行来源分组分析")

    return true_labels, predictions


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
    train_labels, train_predictions = evaluate_dataset(
        args.train_scores, args.train_original, 
        args.log_x, args.threshold, 
        dataset_name="训练集"
    )
    
    # 评估开发集
    dev_labels, dev_predictions = evaluate_dataset(
        args.dev_scores, args.dev_original, 
        args.log_x, args.threshold, 
        dataset_name="开发集"
    )