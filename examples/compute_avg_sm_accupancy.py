import csv

def compute_average_sm_accupancy(csv_file):
    total = 0.0
    count = 0

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Metric Name"].strip() == "sm__warps_active.avg.pct_of_peak_sustained_active":
                try:
                    val = float(row["Metric Value"].strip())
                    total += val
                    count += 1
                except ValueError:
                    continue  # 跳过无效数字

    if count > 0:
        avg = total / count
        print(f"🔍 平均 SM accupancy 利用率: {avg:.2f}%")
    else:
        print("⚠️ 没有找到该指标的任何数据。")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python compute_avg_sm_accupancy.py <ncu导出的csv文件路径>")
    else:
        compute_average_sm_accupancy(sys.argv[1])
