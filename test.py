import json

def compare_submissions(file1_path, file2_path):
    """
    比较两个 submission JSON 文件中的 id 和 label 是否一致。

    Args:
        file1_path (str): 第一个 JSON 文件的路径。 （包含label，但是没有text）
        file2_path (str): 第二个 JSON 文件的路径。 （包含label和text）

    Returns:
        list: 一个包含不一致的 id 的列表。如果为空，则表示所有 id 和 label 都一致。
    """

    mismatched_ids = []

    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
    except FileNotFoundError:
        print("Error: One or both files not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in one or both files.")
        return None

    # 创建一个 id 到 label 的映射，来自第一个文件
    id_to_label = {item['id']: item['label'] for item in data1}

    # 遍历第二个文件，检查 id 和 label 是否一致
    for item in data2:
        item_id = item['id']
        item_label = item['label']

        if item_id in id_to_label:
            if id_to_label[item_id] != item_label:
                mismatched_ids.append(item_id)
        else:
            print(f"Warning: id {item_id} not found in the first file.")  # 如果在第一个文件中找不到对应的 id

    return mismatched_ids

# 使用示例
file1_path = '/Users/jackielyu/Downloads/finalcode/submission.json'
file2_path = '/Users/jackielyu/Downloads/finalcode/submission1.json'

mismatched_ids = compare_submissions(file1_path, file2_path)

if mismatched_ids is None:
    # 发生错误，不执行后续操作
    pass
elif mismatched_ids:
    print("Mismatched ids found:")
    print(mismatched_ids)
else:
    print("All ids and labels match.")
