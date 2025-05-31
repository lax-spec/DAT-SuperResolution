// 全局变量
let selectedFile = null;
let enhancedImageData = null;
const API_BASE_URL = 'http://localhost:5000';

// DOM元素
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const processBtn = document.getElementById('processBtn');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('errorMsg');
const successMsg = document.getElementById('successMsg');
const resultsSection = document.getElementById('resultsSection');
const originalImage = document.getElementById('originalImage');
const enhancedImage = document.getElementById('enhancedImage');
const originalInfo = document.getElementById('originalInfo');
const enhancedInfo = document.getElementById('enhancedInfo');

// 初始化事件监听器
document.addEventListener('DOMContentLoaded', function() {
    // 文件选择事件
    fileInput.addEventListener('change', handleFileSelect);
    
    // 拖拽事件
    uploadSection.addEventListener('dragover', handleDragOver);
    uploadSection.addEventListener('dragleave', handleDragLeave);
    uploadSection.addEventListener('drop', handleDrop);
    
    // 检查后端服务状态
    checkBackendHealth();
});

// 检查后端服务健康状态
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            showSuccess('后端服务已就绪，模型加载成功！');
        } else {
            showError('后端服务异常，请检查服务状态');
        }
    } catch (error) {
        showError('无法连接到后端服务，请确保服务已启动');
        console.error('Backend health check failed:', error);
    }
}

// 处理文件选择
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processSelectedFile(file);
    }
}

// 处理拖拽悬停
function handleDragOver(event) {
    event.preventDefault();
    uploadSection.classList.add('dragover');
}

// 处理拖拽离开
function handleDragLeave(event) {
    event.preventDefault();
    uploadSection.classList.remove('dragover');
}

// 处理文件拖拽
function handleDrop(event) {
    event.preventDefault();
    uploadSection.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            processSelectedFile(file);
        } else {
            showError('请选择有效的图像文件');
        }
    }
}

// 处理选中的文件
function processSelectedFile(file) {
    selectedFile = file;
    
    // 显示文件信息
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    uploadSection.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-check-circle" style="color: #28a745;"></i>
        </div>
        <div class="upload-text">
            已选择文件: ${file.name}
        </div>
        <div style="color: #6c757d; font-size: 0.9em;">
            文件大小: ${fileSize} MB
        </div>
        <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
            <i class="fas fa-folder-open"></i> 重新选择
        </button>
    `;
    
    // 显示处理按钮
    processBtn.style.display = 'inline-block';
    
    // 预览原始图像
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImage.src = e.target.result;
        
        // 创建临时图像来获取尺寸
        const tempImg = new Image();
        tempImg.onload = function() {
            originalInfo.innerHTML = `
                <strong>尺寸:</strong> ${tempImg.width} × ${tempImg.height} 像素<br>
                <strong>文件大小:</strong> ${fileSize} MB
            `;
        };
        tempImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    hideMessages();
}

// 处理图像超分辨率
async function processImage() {
    if (!selectedFile) {
        showError('请先选择一个图像文件');
        return;
    }
    
    // 显示加载状态
    showLoading();
    processBtn.disabled = true;
    hideMessages();
    
    try {
        // 准备FormData
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // 发送请求到后端
        const response = await fetch(`${API_BASE_URL}/enhance`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || '处理失败');
        }
        
        // 获取处理后的图像
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        
        // 显示结果
        enhancedImage.src = imageUrl;
        enhancedImageData = blob;
        
        // 获取增强图像的尺寸信息
        const tempImg = new Image();
        tempImg.onload = function() {
            enhancedInfo.innerHTML = `
                <strong>尺寸:</strong> ${tempImg.width} × ${tempImg.height} 像素<br>
                <strong>放大倍数:</strong> 4倍<br>
                <strong>文件大小:</strong> ${(blob.size / 1024 / 1024).toFixed(2)} MB
            `;
        };
        tempImg.src = imageUrl;
        
        // 显示结果区域
        resultsSection.style.display = 'block';
        showSuccess('图像超分辨率处理完成！');
        
        // 滚动到结果区域
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        console.error('Processing error:', error);
        showError(`处理失败: ${error.message}`);
    } finally {
        hideLoading();
        processBtn.disabled = false;
    }
}

// 下载增强后的图像
function downloadImage() {
    if (!enhancedImageData) {
        showError('没有可下载的图像');
        return;
    }
    
    const url = URL.createObjectURL(enhancedImageData);
    const a = document.createElement('a');
    a.href = url;
    a.download = `enhanced_${selectedFile.name.split('.')[0]}_4x.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showSuccess('图像下载成功！');
}

// 显示加载状态
function showLoading() {
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
}

// 隐藏加载状态
function hideLoading() {
    loading.style.display = 'none';
}

// 显示错误消息
function showError(message) {
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
    successMsg.style.display = 'none';
    
    // 自动隐藏错误消息
    setTimeout(() => {
        errorMsg.style.display = 'none';
    }, 5000);
}

// 显示成功消息
function showSuccess(message) {
    successMsg.textContent = message;
    successMsg.style.display = 'block';
    errorMsg.style.display = 'none';
    
    // 自动隐藏成功消息
    setTimeout(() => {
        successMsg.style.display = 'none';
    }, 3000);
}

// 隐藏所有消息
function hideMessages() {
    errorMsg.style.display = 'none';
    successMsg.style.display = 'none';
}

// 重置应用状态
function resetApp() {
    selectedFile = null;
    enhancedImageData = null;
    processBtn.style.display = 'none';
    resultsSection.style.display = 'none';
    hideLoading();
    hideMessages();
    
    // 重置上传区域
    uploadSection.innerHTML = `
        <div class="upload-icon">
            <i class="fas fa-cloud-upload-alt"></i>
        </div>
        <div class="upload-text">
            拖拽图像文件到此处，或点击按钮选择文件
        </div>
        <input type="file" id="fileInput" class="file-input" accept="image/*">
        <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
            <i class="fas fa-folder-open"></i> 选择图像
        </button>
    `;
    
    // 重新绑定事件
    const newFileInput = document.getElementById('fileInput');
    newFileInput.addEventListener('change', handleFileSelect);
} 