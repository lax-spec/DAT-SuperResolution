from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import io
import base64
from PIL import Image
import os
import sys
import logging
import psutil  # 添加内存监控
import time

# 添加basicsr到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from basicsr.archs.dat_arch import DAT
from basicsr.utils import img2tensor, tensor2img

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperResolutionModel:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """加载训练好的DAT模型"""
        try:
            # 创建DAT模型实例 (4倍超分辨率)
            # 根据计算得出的正确配置: rpe_biases [945,2] = (2*8-1)*(2*32-1) = 15*63 = 945
            # relative_position_index [256,256] = 8*32 = 256
            self.model = DAT(
                upscale=4,
                in_chans=3,
                img_size=64,
                img_range=1.,
                depth=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                expansion_factor=4,
                resi_connection='1conv',
                split_size=[8, 32],  # 正确的配置
                upsampler='pixelshuffle'
            )
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的权重格式
            if 'params_ema' in checkpoint:
                self.model.load_state_dict(checkpoint['params_ema'])
                logger.info("使用 params_ema 权重")
            elif 'params' in checkpoint:
                self.model.load_state_dict(checkpoint['params'])
                logger.info("使用 params 权重")
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                logger.info("使用 state_dict 权重")
            else:
                # 直接是state_dict
                self.model.load_state_dict(checkpoint)
                logger.info("使用直接权重")
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise e
    
    def get_max_image_size(self):
        """根据可用内存动态计算最大图像尺寸"""
        try:
            # 获取可用内存（GB）
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            # 根据可用内存设置最大尺寸
            if available_memory > 8:
                return 1024  # 8GB以上内存
            elif available_memory > 4:
                return 768   # 4-8GB内存
            elif available_memory > 2:
                return 512   # 2-4GB内存
            else:
                return 256   # 2GB以下内存
        except:
            return 512  # 默认值
    
    def enhance_image(self, image):
        """对图像进行超分辨率处理"""
        try:
            # 预处理
            if isinstance(image, np.ndarray):
                # 确保图像是RGB格式
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # 如果是BGR格式，转换为RGB
                    if image.dtype == np.uint8:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            height, width = image.shape[:2]
            available_memory = psutil.virtual_memory().available / (1024**3)
            
            logger.info(f"输入图像: {width}x{height}, 可用内存: {available_memory:.1f}GB")
            
            # 禁用分块处理，直接处理所有图像
            # 但是限制最大尺寸以避免内存问题
            max_size = 800  # 设置合理的最大尺寸
            
            if max(height, width) > max_size:
                # 计算缩放比例
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # 使用高质量插值进行缩放
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                logger.info(f"图像预缩放: {width}x{height} -> {new_width}x{new_height}")
            
            # 转换为tensor
            img_tensor = img2tensor(image, bgr2rgb=False, float32=True)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # 检查模型输出的数值范围
            output_min = output.min().item()
            output_max = output.max().item()
            output_mean = output.mean().item()
            logger.info(f"模型输出范围: min={output_min:.4f}, max={output_max:.4f}, mean={output_mean:.4f}")
            
            # 直接使用tensor2img转换，不进行任何颜色校正
            output_img = tensor2img(output, rgb2bgr=False, min_max=(output_min, output_max))
            
            logger.info(f"=== 纯超分辨率处理完成 ===")
            logger.info(f"输出图像尺寸: {output_img.shape[1]}x{output_img.shape[0]}")
            
            # 清理内存
            del img_tensor, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_img
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            raise e

    def enhance_image_tiled(self, image, tile_size=256, overlap=32):
        """分块处理大图像"""
        try:
            height, width = image.shape[:2]
            logger.info(f"使用分块处理: 图像尺寸 {width}x{height}, 块大小 {tile_size}, 重叠 {overlap}")
            
            # 创建输出图像
            output_height, output_width = height * 4, width * 4
            output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # 计算分块数量
            step = tile_size - overlap
            h_tiles = (height + step - 1) // step
            w_tiles = (width + step - 1) // step
            total_tiles = h_tiles * w_tiles
            
            logger.info(f"总共需要处理 {total_tiles} 个块 ({h_tiles}x{w_tiles})")
            
            processed_tiles = 0
            
            for h_idx in range(h_tiles):
                for w_idx in range(w_tiles):
                    # 计算当前块的位置
                    start_h = h_idx * step
                    start_w = w_idx * step
                    end_h = min(start_h + tile_size, height)
                    end_w = min(start_w + tile_size, width)
                    
                    # 提取当前块
                    tile = image[start_h:end_h, start_w:end_w]
                    
                    # 如果块太小，跳过
                    if tile.shape[0] < 32 or tile.shape[1] < 32:
                        continue
                    
                    # 处理当前块
                    tile_tensor = img2tensor(tile, bgr2rgb=False, float32=True)
                    tile_tensor = tile_tensor.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        tile_output = self.model(tile_tensor)
                    
                    # 检查分块输出范围
                    if processed_tiles == 0:  # 只在第一个块时记录
                        logger.info(f"分块输出范围: min={tile_output.min().item():.4f}, max={tile_output.max().item():.4f}")
                    
                    tile_result = tensor2img(tile_output, rgb2bgr=False)
                    
                    # 计算输出位置
                    out_start_h = start_h * 4
                    out_start_w = start_w * 4
                    out_end_h = min(out_start_h + tile_result.shape[0], output_height)
                    out_end_w = min(out_start_w + tile_result.shape[1], output_width)
                    
                    # 确保不超出边界
                    tile_h = out_end_h - out_start_h
                    tile_w = out_end_w - out_start_w
                    
                    # 放置结果
                    output_image[out_start_h:out_end_h, out_start_w:out_end_w] = tile_result[:tile_h, :tile_w]
                    
                    processed_tiles += 1
                    if processed_tiles % 5 == 0:  # 每5个块报告一次进度
                        progress = (processed_tiles / total_tiles) * 100
                        logger.info(f"处理进度: {progress:.1f}% ({processed_tiles}/{total_tiles})")
                    
                    # 清理内存
                    del tile_tensor, tile_output, tile_result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            logger.info("分块处理完成")
            
            # 应用图像质量增强
            output_image = enhance_image_quality(output_image)
            
            return output_image
            
        except Exception as e:
            logger.error(f"分块处理失败: {e}")
            raise e

# 初始化模型
model_path = "../net_g_150000.pth"
sr_model = SuperResolutionModel(model_path)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy", "model_loaded": sr_model.model is not None})

@app.route('/enhance', methods=['POST'])
def enhance_image():
    """图像超分辨率处理接口"""
    try:
        # 检查请求中是否包含文件
        if 'image' not in request.files:
            return jsonify({"error": "没有上传图像文件"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "没有选择文件"}), 400
        
        # 读取图像
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        
        # 记录原始图像信息
        original_height, original_width = image_np.shape[:2]
        logger.info(f"处理图像: {original_width}x{original_height}")
        
        # 进行超分辨率处理
        enhanced_image = sr_model.enhance_image(image_np)
        
        # 转换为PIL图像
        enhanced_pil = Image.fromarray(enhanced_image)
        
        # 保存到内存
        img_buffer = io.BytesIO()
        enhanced_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # 记录处理后图像信息
        new_height, new_width = enhanced_image.shape[:2]
        logger.info(f"处理完成: {new_width}x{new_height}")
        
        # 调试：保存处理后的图像到本地
        try:
            debug_path = f"debug_enhanced_{int(time.time())}.png"
            cv2.imwrite(debug_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
            logger.info(f"调试图像已保存: {debug_path}")
        except Exception as e:
            logger.warning(f"保存调试图像失败: {e}")
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='enhanced_image.png'
        )
        
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/enhance_base64', methods=['POST'])
def enhance_image_base64():
    """Base64格式的图像超分辨率处理接口"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "没有提供图像数据"}), 400
        
        # 解码base64图像
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        
        # 记录原始图像信息
        original_height, original_width = image_np.shape[:2]
        
        # 进行超分辨率处理
        enhanced_image = sr_model.enhance_image(image_np)
        
        # 转换为base64
        enhanced_pil = Image.fromarray(enhanced_image)
        img_buffer = io.BytesIO()
        enhanced_pil.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # 记录处理后图像信息
        new_height, new_width = enhanced_image.shape[:2]
        
        return jsonify({
            "enhanced_image": f"data:image/png;base64,{img_base64}",
            "original_size": {"width": original_width, "height": original_height},
            "enhanced_size": {"width": new_width, "height": new_height},
            "scale_factor": 4
        })
        
    except Exception as e:
        logger.error(f"处理Base64请求时出错: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 