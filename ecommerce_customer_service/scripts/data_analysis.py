from pathlib import Path
from collections import Counter

def data_analysis():
    """分析清洗后的数据，统计文本长度分布等信息"""
    
    data_dir = Path("ecommerce_customer_service/data/cleaned")
    lengths = []
    max_length = 0
    
    for file_path in data_dir.iterdir():
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_length = len(line.strip())
                    lengths.append(line_length)
                    max_length = max(max_length, line_length)
    
    # 统计长度分布
    length_counts = Counter(lengths)
    
    # 打印统计结果
    print("文本长度分布:")
    print(f"最大长度: {max_length}")
    for length, count in length_counts.most_common():
        print(f"长度 {length}: {count} 行")
    
    
if __name__ == "__main__":
    data_analysis()