# auto-upright
Like lightroom auto upright, automatically detects vertical/horizontal lines in images and corrects perspective distortion (trapezoidal correction).


Auto Upright - 自动透视矫正 JS 库
功能：自动检测图片中的垂直/水平线，修正透视畸变（梯形校正）
OpenCV.js (https://docs.opencv.org/4.x/opencv.js)
 
##example
import { autoUpright, loadOpenCV } from './auto-upright.js';

1. 先加载 OpenCV (只需一次)

  await loadOpenCV();

3. 处理图片
  const result = await autoUpright(imageFile, {

    mode: 'auto',      // 'auto' | 'vertical' | 'full'
   
    autoCrop: true,    // 自动裁切黑边
   
    outputFormat: 'blob'  // 'blob' | 'dataUrl' | 'canvas'
   
  });
