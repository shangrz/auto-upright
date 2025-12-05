# Auto Upright JS

**自动透视矫正库 | Automatic Perspective Correction Library**

一个基于 OpenCV.js 的轻量级图片透视矫正库，自动检测垂直线并修正透视畸变。

A lightweight image perspective correction library powered by OpenCV.js that automatically detects vertical lines and corrects perspective distortion.

---
![截屏2025-12-06 00 19 02](https://github.com/user-attachments/assets/3f7abdd7-6930-4224-8fe7-c7c45f0f1ff1)

---

## 📦 安装 | Installation

### 1. 引入 OpenCV.js

```html
<script async src="https://docs.opencv.org/4.x/opencv.js"></script>
```

### 2. 导入库 | Import Library

```javascript
import { autoUpright, autoCrop } from './auto-upright-x2.js';
```

---

## 🚀 API

### `autoUpright(imageSrc, options)`

自动透视矫正主函数。  
Main function for automatic perspective correction.

**参数 | Parameters:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `imageSrc` | `string` | 图片 URL 或 Data URL |
| `options.autoCrop` | `boolean` | 是否自动裁切黑边 (默认 `true`) |

**返回 | Returns:**

```typescript
Promise<{
  success: boolean;  // 是否成功
  image?: string;    // 矫正后的图片 (Data URL)
  error?: string;    // 错误信息
}>
```

**示例 | Example:**

```javascript
// 等待 OpenCV 加载完成
function waitForOpenCV() {
  return new Promise(resolve => {
    const check = () => {
      if (window.cv && window.cv.Mat) resolve();
      else setTimeout(check, 100);
    };
    check();
  });
}

await waitForOpenCV();

// 处理图片
const result = await autoUpright(imageDataUrl, { autoCrop: true });

if (result.success) {
  document.getElementById('output').src = result.image;
} else {
  console.error(result.error);
}
```

---

### `autoCrop(imageSrc)`

自动裁切图片黑边/透明边框。  
Automatically crops black or transparent borders from an image.

**参数 | Parameters:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `imageSrc` | `string` | 图片 URL 或 Data URL |

**返回 | Returns:**

```typescript
Promise<string>  // 裁切后的图片 Data URL
```

**示例 | Example:**

```javascript
const croppedImage = await autoCrop(originalImageDataUrl);
document.getElementById('output').src = croppedImage;
```

---

## ⚙️ 工作原理 | How It Works

1. **直线检测 | Line Detection**  
   使用 Canny 边缘检测 + Hough 变换检测图片中的直线

2. **垂直线过滤 | Vertical Line Filtering**  
   筛选接近垂直的线段（±25° 容差）

3. **消失点计算 | Vanishing Point Calculation**  
   使用 RANSAC 算法计算垂直线的消失点

4. **透视变换 | Perspective Transform**  
   根据消失点计算单应性矩阵，执行透视矫正

5. **智能裁切 | Smart Cropping**  
   使用最大内切矩形算法裁切黑边

---

## 📐 算法限制 | Limitations

- 需要至少 **4 条垂直线** 才能进行矫正
- 旋转角度限制在 **±15°** 以内
- 输出尺寸不超过原图的 **3 倍**

---

## 🔧 依赖 | Dependencies

- [OpenCV.js 4.x](https://docs.opencv.org/4.x/opencv.js)

---

## 📝 示例文件 | Demo

查看 `auto-upright-demo.html` 获取完整的使用示例。

See `auto-upright-demo.html` for a complete usage example.

---

## 📄 License

MIT
