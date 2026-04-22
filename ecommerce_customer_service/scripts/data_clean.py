import re
import tqdm

def data_clean(input_file: str, output_file: str):
    """去除以"0"开头的行；去除空格（保留\t、\n）"""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"开始清洗文件: {input_file}，总行数: {len(lines)}")
    pbar = tqdm.tqdm(lines, desc="Cleaning")
    cleaned_lines = []
    for line in pbar:
        # 去除以"0"开头的行
        if line.strip().startswith("0"):
            continue
        # 去除空格，保留\t和\n
        cleaned_line = line.replace(" ", "")
        # 去除每行开头的"1\t"
        if cleaned_line.startswith("1\t"):
            cleaned_line = cleaned_line[2:]
        cleaned_lines.append(cleaned_line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"清洗完成，输出文件: {output_file}")
    print(f"处理行数: {len(cleaned_lines)}")


if __name__ == "__main__":
    files = [
        ("ecommerce_customer_service/data/train.txt", "ecommerce_customer_service/data/cleaned/train_cleaned.txt"),
        ("ecommerce_customer_service/data/test.txt", "ecommerce_customer_service/data/cleaned/test_cleaned.txt"),
        ("ecommerce_customer_service/data/dev.txt", "ecommerce_customer_service/data/cleaned/dev_cleaned.txt"),
    ]
    for input_file, output_file in files:
        data_clean(input_file, output_file)
