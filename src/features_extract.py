import json
import os
import argparse
import gc
import numpy as np
import torch
from tqdm import tqdm
import transformers
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 禁用梯度计算
torch.set_grad_enabled(False)

class BinocularsComputer:
    def __init__(
        self,
        observer_model_id="Qwen/Qwen2.5-7B",
        performer_model_id="Qwen/Qwen2.5-7B-Instruct",
        use_bfloat16=True,
        max_token_observed=512,
        compute_mode="both",  # "observer", "performer", "both"
        output_dir="computed_scores"
    ):
        self.observer_model_id = observer_model_id
        self.performer_model_id = performer_model_id
        self.use_bfloat16 = use_bfloat16
        self.max_token_observed = max_token_observed
        self.compute_mode = compute_mode
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 初始化损失函数
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        
        self.observer_model = None
        self.performer_model = None
        self.tokenizer = None
        
    def load_models(self):
        """按需加载模型，以便在处理完后可以释放"""
        print(f"正在加载模型...")
        if self.compute_mode in ["observer", "both"]:
            print(f"加载观察者模型 {self.observer_model_id}...")
            self.observer_model = AutoModelForCausalLM.from_pretrained(
                self.observer_model_id,
                torch_dtype=torch.bfloat16 if self.use_bfloat16 else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.observer_model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.observer_model_id, 
                trust_remote_code=True
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        if self.compute_mode in ["performer", "both"]:
            print(f"加载执行者模型 {self.performer_model_id}...")
            self.performer_model = AutoModelForCausalLM.from_pretrained(
                self.performer_model_id,
                torch_dtype=torch.bfloat16 if self.use_bfloat16 else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            self.performer_model.eval()
            if self.compute_mode == "performer" and self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.performer_model_id, 
                    trust_remote_code=True
                )
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def unload_models(self):
        """释放模型内存"""
        print("正在释放模型内存...")
        if self.observer_model is not None:
            del self.observer_model
            self.observer_model = None
        
        if self.performer_model is not None:
            del self.performer_model
            self.performer_model = None
            
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
    
    def _tokenize(self, batch):
        """将文本转换为token"""
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False
        ).to(self.get_device())
        return encodings
    
    def get_device(self):
        if self.compute_mode == "observer":
            return self.observer_model.device
        elif self.compute_mode == "performer":
            return self.performer_model.device
        else:
            return self.observer_model.device
    
    @torch.inference_mode()
    def _compute_observer_scores(self, encodings):
        """计算观察者模型的logits和perplexity"""
        outputs = self.observer_model(**encodings)
        logits = outputs.logits
        
        # 计算perplexity
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = encodings.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encodings.attention_mask[..., 1:].contiguous()
        
        ce_loss = self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
        ce_loss = (ce_loss * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        
        # 确保转换为float32以防止NumPy转换错误
        return {
            "logits": logits.detach().cpu(),
            "perplexity": ce_loss.detach().cpu().float().numpy(),  # 转为float32
            "input_ids": encodings.input_ids.detach().cpu(),
            "attention_mask": encodings.attention_mask.detach().cpu()
        }
    
    @torch.inference_mode()
    def _compute_performer_scores(self, encodings, observer_logits=None):
        """计算执行者模型的logits和cross-perplexity"""
        outputs = self.performer_model(**encodings)
        logits = outputs.logits
        
        # 计算perplexity
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = encodings.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encodings.attention_mask[..., 1:].contiguous()
        
        ce_loss = self.ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)
        ce_loss = (ce_loss * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        performer_ppl = ce_loss.detach().cpu().float().numpy()  # 转为float32
        
        # 如果有观察者logits，计算cross-perplexity
        cross_ppl = None
        if observer_logits is not None:
            observer_logits = observer_logits.to(self.performer_model.device)
            
            vocab_size = observer_logits.shape[-1]
            total_tokens = logits.shape[-2]
            
            p_proba = self.softmax_fn(observer_logits).view(-1, vocab_size)
            q_scores = logits.view(-1, vocab_size)
            
            ce = self.ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens)
            padding_mask = (encodings.input_ids != self.tokenizer.pad_token_id).type(torch.uint8)
            
            cross_ppl = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).cpu().float().numpy())  # 转为float32
        
        return {
            "logits": logits.detach().cpu(),
            "perplexity": performer_ppl,
            "cross_perplexity": cross_ppl
        }
    
    def process_data(self, data_file, batch_size=4, split_name=None):
        """处理数据文件并计算分数"""
        print(f"正在处理文件: {data_file}")
        
        # 加载数据
        if data_file.endswith('.jsonl'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {data_file}")
        
        # 如果是字典格式，转为列表
        if isinstance(data, dict):
            data = list(data.values())
        
        # 从数据中获取文本
        texts = []
        ids = []
        labels = []
        
        for item in data:
            if 'text' in item:
                texts.append(item['text'])
                
                if 'id' in item:
                    ids.append(item['id'])
                else:
                    ids.append(None)
                    
                if 'label' in item:
                    labels.append(item['label'])
                else:
                    labels.append(None)
        
        # 加载模型
        self.load_models()
        
        # 创建结果数组
        results = []
        
        # 按批次处理
        for i in tqdm(range(0, len(texts), batch_size), desc=f"处理 {split_name or data_file}"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            encodings = self._tokenize(batch_texts)
            
            observer_data = None
            if self.compute_mode in ["observer", "both"]:
                observer_data = self._compute_observer_scores(encodings.to(self.observer_model.device))
            
            performer_data = None
            if self.compute_mode in ["performer", "both"]:
                observer_logits = observer_data["logits"] if observer_data else None
                performer_data = self._compute_performer_scores(
                    encodings.to(self.performer_model.device), 
                    observer_logits
                )
            
            # 合并结果
            for j in range(len(batch_texts)):
                result = {
                    "text": batch_texts[j],
                    "id": batch_ids[j],
                    "label": batch_labels[j]
                }
                
                if observer_data:
                    result["observer_perplexity"] = float(observer_data["perplexity"][j])
                
                if performer_data:
                    result["performer_perplexity"] = float(performer_data["perplexity"][j])
                    
                    if performer_data["cross_perplexity"] is not None:
                        result["cross_perplexity"] = float(performer_data["cross_perplexity"][j])
                        
                        # 计算binoculars score
                        result["binoculars_score"] = result["performer_perplexity"] / result["cross_perplexity"]
                
                results.append(result)
            
            # 定期释放一些内存
            if i % 50 == 0 and i > 0:
                torch.cuda.empty_cache()
        
        # 卸载模型以释放内存
        self.unload_models()
        
        # 确定输出文件名
        if split_name:
            output_file = f"{self.output_dir}/{split_name}_scores.json"
        else:
            base_name = os.path.basename(data_file).split('.')[0]
            output_file = f"{self.output_dir}/{base_name}_scores.json"
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"计算完成，结果已保存到: {output_file}")
        return output_file

def compute_binoculars_scores(train_file, dev_file, test_file, batch_size=4, compute_mode="both", 
                             observer_model="Qwen/Qwen2.5-7B", performer_model="Qwen/Qwen2.5-7B-Instruct",
                             output_dir="/root/autodl-tmp/computed_scores"):
    """计算所有数据集的Binoculars分数"""
    computer = BinocularsComputer(
        observer_model_id=observer_model,
        performer_model_id=performer_model,
        compute_mode=compute_mode,
        output_dir=output_dir
    )
    
    # 逐个处理数据集，处理完一个就卸载模型释放内存
    print("处理训练集...")
    train_output = computer.process_data(train_file, batch_size=batch_size, split_name="train")
    
    print("处理验证集...")
    dev_output = computer.process_data(dev_file, batch_size=batch_size, split_name="dev")
    
    print("处理测试集...")
    test_output = computer.process_data(test_file, batch_size=batch_size, split_name="test")
    
    return {
        "train": train_output,
        "dev": dev_output,
        "test": test_output
    }

def main():
    parser = argparse.ArgumentParser(description='计算观察者和执行者模型的分数')
    parser.add_argument('--train_file', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--dev_file', type=str, required=True, help='验证数据文件路径')
    parser.add_argument('--test_file', type=str, required=True, help='测试数据文件路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批处理大小')
    parser.add_argument('--compute_mode', type=str, default='both', 
                       choices=['observer', 'performer', 'both'], help='计算模式')
    parser.add_argument('--observer_model', type=str, default='Qwen/Qwen2.5-7B', help='观察者模型ID')
    parser.add_argument('--performer_model', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='执行者模型ID')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/computed_scores', help='输出目录')
    
    args = parser.parse_args()
    
    print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 使用优化的处理流程
    outputs = compute_binoculars_scores(
        args.train_file, 
        args.dev_file, 
        args.test_file, 
        batch_size=args.batch_size,
        compute_mode=args.compute_mode,
        observer_model=args.observer_model,
        performer_model=args.performer_model,
        output_dir=args.output_dir
    )
    
    print("所有计算完成！")
    print(f"训练集结果: {outputs['train']}")
    print(f"验证集结果: {outputs['dev']}")
    print(f"测试集结果: {outputs['test']}")

class BinocularsPredictor:
    def __init__(self, score_files, threshold=None):
        """
        使用预先计算的分数进行预测
        
        参数:
        - score_files: 包含预计算分数的文件路径，字典格式，keys为'train', 'dev', 'test'
        - threshold: 分类阈值，如果为None则自动在dev上计算最佳阈值
        """
        self.score_files = score_files
        self.threshold = threshold
        self.scores = self._load_scores()
        
        if threshold is None:
            self.calibrate_threshold()
    
    def _load_scores(self):
        """加载预计算的分数"""
        scores = {}
        for split, file_path in self.score_files.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                scores[split] = json.load(f)
        return scores
    
    def calibrate_threshold(self):
        """在验证集上校准最佳阈值"""
        dev_data = self.scores['dev']
        
        # 提取分数和标签
        scores = []
        labels = []
        
        for item in dev_data:
            if 'binoculars_score' in item and item['label'] is not None:
                scores.append(item['binoculars_score'])
                labels.append(item['label'])
        
        if not scores:
            raise ValueError("无法校准阈值: 没有足够的数据")
        
        # 计算最佳阈值
        best_f1 = 0
        best_threshold = 0
        
        # 排序并获取唯一分数
        sorted_data = sorted(zip(scores, labels), key=lambda x: x[0])
        unique_scores = sorted(list(set(scores)))
        
        for threshold in unique_scores:
            predictions = [1 if score < threshold else 0 for score in scores]
            tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
            fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
            fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"校准完成，最佳阈值: {best_threshold}, 验证集F1: {best_f1}")
        return best_threshold, best_f1
    
    def predict(self, split='test'):
        """在指定数据集上进行预测"""
        if self.threshold is None:
            raise ValueError("必须先校准阈值")
            
        data = self.scores[split]
        predictions = []
        
        for item in data:
            if 'binoculars_score' in item:
                pred = 1 if item['binoculars_score'] < self.threshold else 0
                predictions.append({
                    'id': item['id'],
                    'prediction': pred
                })
        
        return predictions
    
    def evaluate(self, split='test'):
        """评估在指定数据集上的性能"""
        data = self.scores[split]
        
        # 提取分数和标签
        scores = []
        labels = []
        
        for item in data:
            if 'binoculars_score' in item and item['label'] is not None:
                scores.append(item['binoculars_score'])
                labels.append(item['label'])
        
        if not labels:
            raise ValueError(f"无法评估: {split}集没有标签")
        
        # 使用阈值进行预测
        predictions = [1 if score < self.threshold else 0 for score in scores]
        
        # 计算指标
        tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
        fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
        fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
        tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
        
        accuracy = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': self.threshold
        }
    
    def save_predictions(self, output_file, split='test'):
        """保存预测结果"""
        predictions = self.predict(split)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')
        
        print(f"预测结果已保存到: {output_file}")
        return output_file

def predict_from_scores(train_score_file, dev_score_file, test_score_file, output_file):
    """从预计算的分数文件进行预测"""
    predictor = BinocularsPredictor({
        'train': train_score_file,
        'dev': dev_score_file,
        'test': test_score_file
    })
    
    # 评估在验证集上的性能
    dev_metrics = predictor.evaluate('dev')
    print("验证集评估结果:")
    for metric, value in dev_metrics.items():
        print(f"  {metric}: {value}")
    
    # 保存测试集预测结果
    predictor.save_predictions(output_file)
    
    # 如果测试集有标签，也评估其性能
    try:
        test_metrics = predictor.evaluate('test')
        print("测试集评估结果:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value}")
    except ValueError:
        print("测试集没有标签，跳过评估")

if __name__ == "__main__":
    main()