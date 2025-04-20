import json
import argparse

def json_to_jsonl(json_file, jsonl_file, remove_newlines=True):
    """
    将JSON文件转换为JSONL格式，并可选择去除text字段中的换行符
    
    参数:
        json_file: 输入的JSON文件路径
        jsonl_file: 输出的JSONL文件路径
        remove_newlines: 是否去除text字段中的换行符，默认为True
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 去除text字段中的换行符
            if remove_newlines and 'text' in item:
                item['text'] = item['text'].replace('\n', ' ')
            
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已将 {json_file} 转换为 {jsonl_file}")
    if remove_newlines:
        print("已去除所有text字段中的换行符")

def process_json(json_file, output_file, remove_newlines=True):
    """
    处理JSON文件，去除text字段中的换行符并保存
    
    参数:
        json_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径
        remove_newlines: 是否去除text字段中的换行符，默认为True
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if remove_newlines:
        for item in data:
            if 'text' in item:
                item['text'] = item['text'].replace('\n', ' ')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"已处理 {json_file} 并保存到 {output_file}")
    if remove_newlines:
        print("已去除所有text字段中的换行符")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理JSON/JSONL文件并去除text字段中的换行符')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--format', type=str, choices=['json', 'jsonl'], default='jsonl', help='输出格式，json或jsonl')
    parser.add_argument('--keep-newlines', action='store_false', dest='remove_newlines', help='保留换行符（默认是去除）')
    
    args = parser.parse_args()
    
    if args.format == 'jsonl':
        json_to_jsonl(args.input, args.output, args.remove_newlines)
    else:
        process_json(args.input, args.output, args.remove_newlines)