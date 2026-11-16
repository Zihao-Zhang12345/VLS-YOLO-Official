from ultralytics import YOLO
import argparse
import os
# --- 新增的导入 ---
import numpy as np
from prettytable import PrettyTable
from ultralytics.utils.torch_utils import model_info
# --- 结束 ---


ROOT = os.path.abspath('.') + "/"


# --- 新增的辅助函数 ---
def get_weight_size(model_path):
    """获取模型文件的大小 (MB)"""
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f}"
    return "N/A"
# --- 结束 ---


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + 'ultralytics/cfg/datasets/VOC-test.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT + 'output_dir/tiaozheng/test-1312/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--batch', type=int, default=10, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT + 'output_dir/xss', help='save to project/name')
    parser.add_argument('--name', default='val-last', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    # 加载训练好的模型权重
    model = YOLO(opt.weights)

    # 在验证集上运行评估
    results = model.val(
        data=opt.data,
        batch=opt.batch,
        imgsz=opt.imgsz,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        half=opt.half,
        dnn=opt.dnn
    )
    
    # =================================================================
    # ---               新增的结果记录脚本 (开始)                   ---
    # =================================================================
    if model.task == 'detect': # 仅目标检测任务适用
        # 在 val.py 中，results 就是最终的验证结果对象
        result = results
        model_path = opt.weights # 模型路径就是传入的 weights 参数

        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                model_names[idx], 
                f"{result.box.p[idx]:.4f}", 
                f"{result.box.r[idx]:.4f}", 
                f"{result.box.f1[idx]:.4f}", 
                f"{result.box.ap50[idx]:.4f}", 
                f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                f"{result.box.ap[idx]:.4f}"
            ])
        model_metrice_table.add_row([
            "all(平均数据)", 
            f"{result.results_dict['metrics/precision(B)']:.4f}", 
            f"{result.results_dict['metrics/recall(B)']:.4f}", 
            f"{np.mean(result.box.f1[:length]):.4f}", 
            f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
            f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
    # =================================================================
    # ---               新增的结果记录脚本 (结束)                   ---
    # =================================================================