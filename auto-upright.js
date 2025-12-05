/**
 * Auto Upright - 自动透视矫正 JS 库
 * 
 * 功能：自动检测图片中的垂直/水平线，修正透视畸变（梯形校正）
 * 依赖：OpenCV.js (https://docs.opencv.org/4.x/opencv.js)
 * 
 * @example
 * import { autoUpright, loadOpenCV } from './auto-upright.js';
 * 
 * // 1. 先加载 OpenCV (只需一次)
 * await loadOpenCV();
 * 
 * // 2. 处理图片
 * const result = await autoUpright(imageFile, {
 *   mode: 'auto',      // 'auto' | 'vertical' | 'full'
 *   autoCrop: true,    // 自动裁切黑边
 *   outputFormat: 'blob'  // 'blob' | 'dataUrl' | 'canvas'
 * });
 */

const OPENCV_CDN = 'https://docs.opencv.org/4.x/opencv.js';

// ============================================
// OpenCV 加载器
// ============================================

let cvReady = false;
let cvPromise = null;

/**
 * 加载 OpenCV.js（只需调用一次）
 * @returns {Promise<void>}
 */
export async function loadOpenCV(customUrl) {
    if (cvReady && window.cv?.Mat) return;

    if (cvPromise) return cvPromise;

    cvPromise = new Promise((resolve, reject) => {
        if (window.cv?.Mat) {
            cvReady = true;
            resolve();
            return;
        }

        const script = document.createElement('script');
        script.src = customUrl || OPENCV_CDN;
        script.async = true;

        script.onload = () => {
            const cv = window.cv;
            if (!cv) {
                reject(new Error('OpenCV.js loaded but cv is undefined'));
                return;
            }
            if (cv.Mat) {
                cvReady = true;
                resolve();
                return;
            }
            cv.onRuntimeInitialized = () => {
                cvReady = true;
                resolve();
            };
        };

        script.onerror = () => reject(new Error('Failed to load OpenCV.js'));
        document.head.appendChild(script);
    });

    return cvPromise;
}

/**
 * 检查 OpenCV 是否就绪
 */
export function isOpenCVReady() {
    return cvReady && !!window.cv?.Mat;
}

// ============================================
// 数学工具函数
// ============================================

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const ang180 = (rad) => {
    let d = (rad * 180) / Math.PI;
    return ((d % 180) + 180) % 180;
};
const angDiff = (a, b) => {
    let d = Math.abs(a - b) % 180;
    return d > 90 ? 180 - d : d;
};
const mod180 = (a) => ((a % 180) + 180) % 180;
const mod180S = (a) => {
    let r = mod180(a);
    return r > 90 ? r - 180 : r;
};

const line2 = (x1, y1, x2, y2) => [y1 - y2, x2 - x1, x1 * y2 - x2 * y1];
const normL = (l) => {
    const s = Math.hypot(l[0], l[1]) || 1;
    return [l[0] / s, l[1] / s, l[2] / s];
};

const inter = (l1, l2) => {
    const det = l1[0] * l2[1] - l2[0] * l1[1];
    if (Math.abs(det) < 1e-12) return null;
    return [(l1[1] * l2[2] - l2[1] * l1[2]) / det, (l1[2] * l2[0] - l2[2] * l1[0]) / det];
};

const distLinePoint = (l, x, y) => {
    const d = Math.hypot(l[0], l[1]);
    return d < 1e-12 ? 1e9 : Math.abs(l[0] * x + l[1] * y + l[2]) / d;
};

const inv2 = (m) => {
    const det = m[0] * m[3] - m[1] * m[2];
    if (Math.abs(det) < 1e-12) return null;
    const k = 1 / det;
    return [m[3] * k, -m[1] * k, -m[2] * k, m[0] * k];
};

const mul3 = (A, B) => {
    const C = new Array(9).fill(0);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[c + 3] + A[r * 3 + 2] * B[c + 6];
        }
    }
    return C;
};

const appH = (H, x, y) => [
    H[0] * x + H[1] * y + H[2],
    H[3] * x + H[4] * y + H[5],
    H[6] * x + H[7] * y + H[8]
];

function safeXY(H, x, y) {
    const [X, Y, W] = appH(H, x, y);
    if (!Number.isFinite(X) || !Number.isFinite(Y) || !Number.isFinite(W) || Math.abs(W) < 1e-12) return null;
    return [X / W, Y / W];
}

// ============================================
// 核心算法
// ============================================

function distortionScore(H, w, h) {
    const grid = 5, eps = 1;
    let sumLogCond = 0, sumAbsLogArea = 0, n = 0;
    for (let gy = 1; gy <= grid; gy++) {
        for (let gx = 1; gx <= grid; gx++) {
            const x = (gx / (grid + 1)) * w, y = (gy / (grid + 1)) * h;
            const p = safeXY(H, x, y), px = safeXY(H, x + eps, y), py = safeXY(H, x, y + eps);
            if (!p || !px || !py) continue;
            const j11 = (px[0] - p[0]) / eps, j21 = (px[1] - p[1]) / eps;
            const j12 = (py[0] - p[0]) / eps, j22 = (py[1] - p[1]) / eps;
            const a = j11 * j11 + j21 * j21, b = j11 * j12 + j21 * j22, c = j12 * j12 + j22 * j22;
            const tr = a + c, det = a * c - b * b;
            if (!(det > 0) || !(tr > 0)) continue;
            const disc = Math.max(0, tr * tr - 4 * det);
            const l1 = (tr + Math.sqrt(disc)) / 2, l2 = (tr - Math.sqrt(disc)) / 2;
            if (!(l1 > 0) || !(l2 > 0)) continue;
            const cond = Math.sqrt(l1) / Math.sqrt(l2);
            const area = Math.abs(j11 * j22 - j12 * j21);
            if (!Number.isFinite(cond) || !Number.isFinite(area) || area <= 1e-12) continue;
            sumLogCond += Math.log(cond);
            sumAbsLogArea += Math.abs(Math.log(area));
            n++;
        }
    }
    return n <= 0 ? 9e9 : sumLogCond / n + 0.7 * (sumAbsLogArea / n);
}

function bbox(H, w, h) {
    const pts = [[0, 0], [w, 0], [w, h], [0, h]].map(([x, y]) => safeXY(H, x, y));
    if (pts.some(p => !p)) return null;
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const outW = clamp(Math.ceil(maxX - minX), 64, 6000);
    const outH = clamp(Math.ceil(maxY - minY), 64, 6000);
    return { H: mul3([1, 0, -minX, 0, 1, -minY, 0, 0, 1], H), outW, outH };
}

function evalCandidate(H, vSegs, hSegs, w, h) {
    const evalSegs = (segs, target) => {
        let sw = 0, sd = 0;
        for (const s of segs) {
            const p1 = safeXY(H, s.x1, s.y1), p2 = safeXY(H, s.x2, s.y2);
            if (!p1 || !p2) continue;
            const ang = ang180(Math.atan2(p2[1] - p1[1], p2[0] - p1[0]));
            const wgt = Math.max(1e-3, s.len);
            sw += wgt;
            sd += wgt * clamp(angDiff(ang, target), 0, 45);
        }
        return sw > 1e-9 ? sd / sw : 45;
    };
    const dist = distortionScore(H, w, h);
    const bb = bbox(H, w, h);
    return {
        alignV: evalSegs(vSegs, 90),
        alignH: evalSegs(hSegs, 0),
        dist,
        areaRatio: bb ? (bb.outW * bb.outH) / (w * h) : 99,
        proj: Math.hypot(H[6], H[7])
    };
}

function makeRng(seed) {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let x = Math.imul(t ^ (t >>> 15), 1 | t);
        x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
}

function kmeansAxes(segs) {
    if (segs.length < 8) return null;
    const it = segs.slice().sort((a, b) => b.len - a.len).slice(0, 200).map(s => {
        const t = (s.ang * Math.PI) / 180;
        return { w: Math.max(1e-3, s.len), x: Math.cos(2 * t), y: Math.sin(2 * t) };
    });
    const nrm = v => { const n = Math.hypot(v[0], v[1]) || 1; return [v[0] / n, v[1] / n]; };
    let c1 = nrm([it[0].x, it[0].y]);
    let c2 = nrm([it[Math.floor(it.length / 2)].x, it[Math.floor(it.length / 2)].y]);
    for (let k = 0; k < 8; k++) {
        let a = [0, 0, 0], b = [0, 0, 0];
        for (const p of it) {
            const d1 = (p.x - c1[0]) ** 2 + (p.y - c1[1]) ** 2;
            const d2 = (p.x - c2[0]) ** 2 + (p.y - c2[1]) ** 2;
            const t = d1 <= d2 ? a : b;
            t[0] += p.w * p.x; t[1] += p.w * p.y; t[2] += p.w;
        }
        if (a[2] > 1e-9) c1 = nrm([a[0] / a[2], a[1] / a[2]]);
        if (b[2] > 1e-9) c2 = nrm([b[0] / b[2], b[1] / b[2]]);
    }
    const toDeg = c => { let th = Math.atan2(c[1], c[0]) * 0.5; return th < 0 ? (th + Math.PI) * 180 / Math.PI : th * 180 / Math.PI; };
    let A = toDeg(c1), B = toDeg(c2);
    if (angDiff(A, B) < 25) B = (A + 90) % 180;
    return { A, B };
}

function ransacVP(lines, rng, iters = 420, thr = 0.025, minIn = 8) {
    if (lines.length < 2) return null;
    const L = lines.map(normL);
    let best = null;
    for (let t = 0; t < iters; t++) {
        const i = (rng() * L.length) | 0;
        let j = (rng() * L.length) | 0;
        if (j === i) j = (j + 1) % L.length;
        const p = inter(L[i], L[j]);
        if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) continue;
        let c = 0;
        for (const l of L) if (distLinePoint(l, p[0], p[1]) < thr) c++;
        if (!best || c > best.c) best = { x: p[0], y: p[1], c };
    }
    if (!best || best.c < minIn) return null;
    return { x: best.x, y: best.y, in: best.c, total: L.length, conf: best.c / L.length };
}

function rectH(vpH, vpV, w, h) {
    let a = vpH[1] - vpV[1], b = vpV[0] - vpH[0], c = vpH[0] * vpV[1] - vpV[0] * vpH[1];
    if (Math.abs(c) < 1e-12) { const s = Math.hypot(a, b) || 1; a /= s; b /= s; c = 1; }
    else { const s = 1 / c; a *= s; b *= s; c = 1; }
    const Hproj = [1, 0, 0, 0, 1, 0, a, b, 1];
    const vph = appH(Hproj, vpH[0], vpH[1]), vpv = appH(Hproj, vpV[0], vpV[1]);
    let dx = [vph[0], vph[1]], dy = [vpv[0], vpv[1]];
    const ndx = Math.hypot(dx[0], dx[1]) || 1, ndy = Math.hypot(dy[0], dy[1]) || 1;
    dx = [dx[0] / ndx, dx[1] / ndx]; dy = [dy[0] / ndy, dy[1] / ndy];
    const invM = inv2([dx[0], dy[0], dx[1], dy[1]]);
    if (!invM) return null;
    let A = invM;
    if (dx[0] * dy[1] - dy[0] * dx[1] < 0) A = [-A[0], -A[1], A[2], A[3]];
    const Haff = [A[0], A[1], 0, A[2], A[3], 0, 0, 0, 1];
    const cx = w / 2, cy = h / 2;
    return mul3([1, 0, cx, 0, 1, cy, 0, 0, 1], mul3(mul3(Haff, Hproj), [1, 0, -cx, 0, 1, -cy, 0, 0, 1]));
}

function rotC(deg, w, h) {
    const t = (deg * Math.PI) / 180, ct = Math.cos(t), st = Math.sin(t);
    const cx = w / 2, cy = h / 2;
    return mul3([1, 0, cx, 0, 1, cy, 0, 0, 1], mul3([ct, -st, 0, st, ct, 0, 0, 0, 1], [1, 0, -cx, 0, 1, -cy, 0, 0, 1]));
}

function onePointH(vp, w, h) {
    const cx = w / 2, cy = h / 2, vx = vp[0] - cx, vy = vp[1] - cy;
    if (Math.hypot(vx, vy) < 0.28 * Math.max(w, h)) return null;
    const den = vx * vx + vy * vy;
    if (den < 1e-6) return null;
    const P = [1, 0, 0, 0, 1, 0, -vx / den, -vy / den, 1];
    return mul3([1, 0, cx, 0, 1, cy, 0, 0, 1], mul3(P, [1, 0, -cx, 0, 1, -cy, 0, 0, 1]));
}

function rotateThenOnePoint(vp, roll, w, h) {
    const R = rotC(roll, w, h);
    const pr = appH(R, vp[0], vp[1]);
    if (Math.abs(pr[2]) < 1e-12) return R;
    const H1 = onePointH([pr[0] / pr[2], pr[1] / pr[2]], w, h);
    return H1 ? mul3(H1, R) : R;
}

function vpPlausible(vp, kind) {
    const ax = Math.abs(vp.x), ay = Math.abs(vp.y);
    return kind === 'v' ? (ay > 0.55 || ay > ax * 1.2) : (ax > 0.55 || ax > ay * 1.2);
}

function vpConfOK(vp, kind) {
    if (!vp) return false;
    return vp.in >= Math.max(6, Math.floor(vp.total * 0.25)) && vp.conf >= 0.22 && vpPlausible(vp, kind);
}

function candidateSane(H, vSegs, hSegs, w, h) {
    const m = evalCandidate(H, vSegs, hSegs, w, h);
    return Number.isFinite(m.dist) && m.dist <= 3.5 &&
        Number.isFinite(m.areaRatio) && m.areaRatio <= 3.2 &&
        Number.isFinite(m.proj) && m.proj <= 0.02;
}

// ============================================
// 智能内切裁切 (最大内切矩形)
// ============================================

function largestInscribedRect(maskData, rows, cols) {
    const heights = new Uint16Array(rows * cols);
    for (let c = 0; c < cols; c++) heights[c] = maskData[c] > 127 ? 1 : 0;
    for (let r = 1; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const idx = r * cols + c;
            heights[idx] = maskData[idx] > 127 ? heights[(r - 1) * cols + c] + 1 : 0;
        }
    }

    let bestArea = 0, bestRect = null;
    for (let r = 0; r < rows; r++) {
        const stack = [];
        for (let i = 0; i <= cols; i++) {
            const h = i < cols ? heights[r * cols + i] : 0;
            let startIdx = i;
            while (stack.length && stack[stack.length - 1].h > h) {
                const top = stack.pop();
                const area = top.h * (i - top.idx);
                if (area > bestArea) {
                    bestArea = area;
                    bestRect = { x: top.idx, y: r - top.h + 1, w: i - top.idx, h: top.h };
                }
                startIdx = top.idx;
            }
            stack.push({ idx: startIdx, h });
        }
    }
    return bestRect;
}

// ============================================
// 主入口函数
// ============================================

/**
 * 自动透视矫正
 * 
 * @param {File|Blob|HTMLImageElement|HTMLCanvasElement|string} input - 输入图片
 * @param {Object} options - 配置选项
 * @param {string} options.mode - 矫正模式: 'auto' | 'vertical' | 'full' (默认 'auto')
 * @param {boolean} options.autoCrop - 是否自动裁切黑边 (默认 true)
 * @param {boolean} options.blackCorners - 黑色填充边角 (默认 true)
 * @param {string} options.outputFormat - 输出格式: 'blob' | 'dataUrl' | 'canvas' (默认 'blob')
 * @param {string} options.mimeType - 输出 MIME 类型 (默认 'image/png')
 * @param {number} options.quality - JPEG 质量 0-1 (默认 0.92)
 * 
 * @returns {Promise<Blob|string|HTMLCanvasElement>} 矫正后的图片
 */
export async function autoUpright(input, options = {}) {
    const {
        mode = 'auto',
        autoCrop = true,
        blackCorners = true,
        outputFormat = 'blob',
        mimeType = 'image/png',
        quality = 0.92
    } = options;

    if (!isOpenCVReady()) {
        throw new Error('OpenCV not ready. Call loadOpenCV() first.');
    }

    const cv = window.cv;

    // 1. 解码输入图片到 Canvas
    const srcCanvas = await inputToCanvas(input);
    const W0 = srcCanvas.width, H0 = srcCanvas.height;

    // 2. 降采样用于分析
    const maxSide = 900;
    const sf = Math.min(1, maxSide / Math.max(W0, H0));
    const aw = Math.max(2, Math.round(W0 * sf));
    const ah = Math.max(2, Math.round(H0 * sf));

    const analysisCanvas = document.createElement('canvas');
    analysisCanvas.width = aw;
    analysisCanvas.height = ah;
    analysisCanvas.getContext('2d').drawImage(srcCanvas, 0, 0, aw, ah);

    const S = Math.max(aw, ah), cx = aw / 2, cy = ah / 2;
    const toN = (x, y) => [(x - cx) / S, (y - cy) / S];

    // 3. 检测直线
    let src = null, gray = null, blur = null, edges = null, lines = null;
    const del = m => { try { m?.delete?.(); } catch { } };

    try {
        src = cv.imread(analysisCanvas);
        gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        blur = new cv.Mat();
        cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 1.2, 1.2, cv.BORDER_DEFAULT);
        edges = new cv.Mat();
        cv.Canny(blur, edges, 70, 200);
        lines = new cv.Mat();
        const minLen = Math.round(Math.min(aw, ah) * 0.1);
        cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 60, minLen, Math.round(Math.min(aw, ah) * 0.02));

        const segs = [];
        for (let i = 0; i < lines.rows; i++) {
            const x1 = lines.data32S[i * 4], y1 = lines.data32S[i * 4 + 1];
            const x2 = lines.data32S[i * 4 + 2], y2 = lines.data32S[i * 4 + 3];
            const len = Math.hypot(x2 - x1, y2 - y1);
            if (len >= minLen) {
                segs.push({ x1, y1, x2, y2, ang: ang180(Math.atan2(y2 - y1, x2 - x1)), len });
            }
        }

        if (segs.length < 10) {
            // 直线不足，返回原图
            return formatOutput(srcCanvas, outputFormat, mimeType, quality);
        }

        // 4. 计算主轴和分类
        const axes = kmeansAxes(segs) || { A: 0, B: 90 };
        const tol = 26;
        const countHV = r0 => {
            let hc = 0, vc = 0;
            for (const s of segs) {
                const ar = mod180(s.ang + r0);
                if (angDiff(ar, 0) <= tol) hc++;
                if (angDiff(ar, 90) <= tol) vc++;
            }
            return { hc, vc };
        };

        const cands = [0, -axes.A, -(axes.A - 90), -axes.B, -(axes.B - 90)].map(mod180S);
        let bestR = 0, bestScore = -1;
        const hv0 = countHV(0);
        for (const r0 of cands) {
            const { hc, vc } = countHV(r0);
            if (hc < 6 || vc < 6) continue;
            const score = Math.min(hc, vc) * 4 + (hc + vc) - 0.15 * Math.abs(r0);
            if (score > bestScore) { bestScore = score; bestR = r0; }
        }

        const roll = hv0.hc >= 10 && hv0.vc >= 10 ? 0 : bestScore < 0 ? 0 : bestR;
        const rollSmall = Math.abs(roll) <= 8 ? roll : 0;

        const hSegs = segs.filter(s => angDiff(mod180(s.ang + roll), 0) <= tol);
        const vSegs = segs.filter(s => angDiff(mod180(s.ang + roll), 90) <= tol);

        if (hSegs.length < 6 || vSegs.length < 6) {
            return formatOutput(srcCanvas, outputFormat, mimeType, quality);
        }

        // 5. RANSAC 消失点
        const vLines = vSegs.map(s => { const [a1, b1] = toN(s.x1, s.y1), [a2, b2] = toN(s.x2, s.y2); return line2(a1, b1, a2, b2); });
        const hLines = hSegs.map(s => { const [a1, b1] = toN(s.x1, s.y1), [a2, b2] = toN(s.x2, s.y2); return line2(a1, b1, a2, b2); });

        const rng = makeRng(((aw * 73856093) ^ (ah * 19349663) ^ (segs.length * 83492791)) >>> 0);
        const vpV = ransacVP(vLines, rng, 520, 0.025, Math.max(8, Math.floor(vLines.length * 0.2)));
        const vpH = ransacVP(hLines, rng, 520, 0.025, Math.max(8, Math.floor(hLines.length * 0.2)));

        const okV = vpConfOK(vpV, 'v'), okH = vpConfOK(vpH, 'h');
        const vpVpx = okV ? [vpV.x * S + cx, vpV.y * S + cy] : null;
        const vpHpx = okH ? [vpH.x * S + cx, vpH.y * S + cy] : null;

        // 6. 构建候选变换
        const HA_rot = rotC(rollSmall, aw, ah);
        const HA_v = vpVpx ? rotateThenOnePoint(vpVpx, rollSmall, aw, ah) : null;
        let HA_full = null;
        if (vpVpx && vpHpx) {
            const R = rotC(rollSmall, aw, ah);
            const pV = safeXY(R, vpVpx[0], vpVpx[1]), pH = safeXY(R, vpHpx[0], vpHpx[1]);
            if (pV && pH) {
                const H2 = rectH(pH, pV, aw, ah);
                if (H2) HA_full = mul3(H2, R);
            }
        }

        // 7. 模式选择
        let chosenMode = 'rot';
        if (mode === 'vertical') {
            chosenMode = HA_v && candidateSane(HA_v, vSegs, hSegs, aw, ah) ? 'v' : 'rot';
        } else if (mode === 'full') {
            if (HA_full && candidateSane(HA_full, vSegs, hSegs, aw, ah)) chosenMode = 'full';
            else if (HA_v && candidateSane(HA_v, vSegs, hSegs, aw, ah)) chosenMode = 'v';
        } else { // auto
            if (HA_v && candidateSane(HA_v, vSegs, hSegs, aw, ah)) {
                chosenMode = 'v';
                if (HA_full && candidateSane(HA_full, vSegs, hSegs, aw, ah)) {
                    const mv = evalCandidate(HA_v, vSegs, hSegs, aw, ah);
                    const mf = evalCandidate(HA_full, vSegs, hSegs, aw, ah);
                    if (mv.alignH - mf.alignH > 6 && mf.dist - mv.dist < 0.35) chosenMode = 'full';
                }
            }
        }

        let H_A = HA_rot;
        if (chosenMode === 'full' && HA_full) H_A = HA_full;
        else if (chosenMode === 'v' && HA_v) H_A = HA_v;

        // 8. 应用变换到全分辨率
        const Sdown = [sf, 0, 0, 0, sf, 0, 0, 0, 1];
        const Sup = [1 / sf, 0, 0, 0, 1 / sf, 0, 0, 0, 1];
        const H_full = mul3(Sup, mul3(H_A, Sdown));
        let bbFull = bbox(H_full, W0, H0);

        if (!bbFull || (bbFull.outW * bbFull.outH) / (W0 * H0) > 3.2) {
            const bbRot = bbox(mul3(Sup, mul3(HA_rot, Sdown)), W0, H0);
            if (bbRot) bbFull = bbRot;
        }

        if (!bbFull) {
            return formatOutput(srcCanvas, outputFormat, mimeType, quality);
        }

        // 9. 透视变换
        const srcMat = cv.imread(srcCanvas);
        let dstMat = new cv.Mat();
        const M = cv.matFromArray(3, 3, cv.CV_64F, bbFull.H);
        const border = blackCorners ? new cv.Scalar(0, 0, 0, 255) : new cv.Scalar(0, 0, 0, 0);
        cv.warpPerspective(srcMat, dstMat, M, new cv.Size(bbFull.outW, bbFull.outH), cv.INTER_LINEAR, cv.BORDER_CONSTANT, border);

        // 10. 智能裁切
        if (autoCrop) {
            try {
                const maskSrc = new cv.Mat(H0, W0, cv.CV_8UC1, new cv.Scalar(255));
                const maskDst = new cv.Mat();
                cv.warpPerspective(maskSrc, maskDst, M, new cv.Size(bbFull.outW, bbFull.outH), cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));

                const bestRect = largestInscribedRect(maskDst.data, maskDst.rows, maskDst.cols);
                maskDst.delete();
                maskSrc.delete();

                if (bestRect && bestRect.w > 8 && bestRect.h > 8) {
                    const rect = new cv.Rect(bestRect.x, bestRect.y, bestRect.w, bestRect.h);
                    const roi = dstMat.roi(rect);
                    const cropped = roi.clone();
                    roi.delete();
                    dstMat.delete();
                    dstMat = cropped;
                }
            } catch { }
        }

        M.delete();
        srcMat.delete();

        // 11. 输出
        const outCanvas = document.createElement('canvas');
        outCanvas.width = dstMat.cols;
        outCanvas.height = dstMat.rows;
        cv.imshow(outCanvas, dstMat);
        dstMat.delete();

        return formatOutput(outCanvas, outputFormat, mimeType, quality);

    } finally {
        del(src); del(gray); del(blur); del(edges); del(lines);
    }
}

// ============================================
// 辅助函数
// ============================================

async function inputToCanvas(input) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (input instanceof HTMLCanvasElement) {
        canvas.width = input.width;
        canvas.height = input.height;
        ctx.drawImage(input, 0, 0);
        return canvas;
    }

    if (input instanceof HTMLImageElement) {
        canvas.width = input.naturalWidth || input.width;
        canvas.height = input.naturalHeight || input.height;
        ctx.drawImage(input, 0, 0);
        return canvas;
    }

    // File, Blob, or URL string
    const img = await loadImage(input);
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
    return canvas;
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load image'));

        if (src instanceof Blob || src instanceof File) {
            const url = URL.createObjectURL(src);
            img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
            img.src = url;
        } else {
            img.src = src;
        }
    });
}

function formatOutput(canvas, format, mimeType, quality) {
    if (format === 'canvas') return canvas;
    if (format === 'dataUrl') return canvas.toDataURL(mimeType, quality);

    return new Promise(resolve => {
        canvas.toBlob(blob => resolve(blob), mimeType, quality);
    });
}

// ============================================
// 默认导出
// ============================================

export default {
    loadOpenCV,
    isOpenCVReady,
    autoUpright
};
