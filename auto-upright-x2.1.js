/**
 * AutoUpright.js Library
 * 
 * A standalone library for automatic image perspective correction (vertical straightening)
 * using OpenCV.js.
 * 
 * Dependencies:
 * - OpenCV.js (must be loaded globally as window.cv or passed in)
 */

// --- Math Helpers (Internal) ---

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// Standardize angle to [0, 180)
const ang180 = (rad) => {
    let d = (rad * 180) / Math.PI;
    d = ((d % 180) + 180) % 180;
    return d;
};

// Create line equation ax + by + c = 0 from two points
const line2 = (x1, y1, x2, y2) => {
    const a = y1 - y2,
        b = x2 - x1,
        c = x1 * y2 - x2 * y1;
    return [a, b, c];
};

// Distance between lines (Projective)
const dist = (l, x, y) => {
    const [a, b, c] = l;
    const d = Math.hypot(a, b);
    return d < 1e-12 ? 1e9 : Math.abs(a * x + b * y + c) / d;
};

// Intersection of two lines
const inter = (l1, l2) => {
    const [a1, b1, c1] = l1,
        [a2, b2, c2] = l2;
    const det = a1 * b2 - a2 * b1;
    if (Math.abs(det) < 1e-12) return null;
    return [(b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det];
};

// Normalize line equation
const normL = (l) => {
    let [a, b, c] = l;
    const s = Math.hypot(a, b) || 1;
    return [a / s, b / s, c / s];
};

// Matrix multiplication 3x3
const mul3 = (A, B) => {
    const C = new Array(9).fill(0);
    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[c + 3] + A[r * 3 + 2] * B[c + 6];
        }
    }
    return C;
};

// Apply Homography to point
const appH = (H, x, y) => {
    const X = H[0] * x + H[1] * y + H[2],
        Y = H[3] * x + H[4] * y + H[5],
        W = H[6] * x + H[7] * y + H[8];
    return [X, Y, W];
};

// Rotation Matrix
function rotC(deg, w, h) {
    const t = (deg * Math.PI) / 180,
        ct = Math.cos(t),
        st = Math.sin(t);
    const R = [ct, -st, 0, st, ct, 0, 0, 0, 1];
    const cx = w / 2,
        cy = h / 2;
    const T1 = [1, 0, -cx, 0, 1, -cy, 0, 0, 1],
        T2 = [1, 0, cx, 0, 1, cy, 0, 0, 1];
    return mul3(T2, mul3(R, T1));
}

// Generate correction matrix for a single vanishing point (Vertical Upright)
function onePointVerticalH(vp, w, h) {
    const cx = w / 2,
        cy = h / 2;

    // Vanishing point relative to center
    const vx = vp[0] - cx;
    const vy = vp[1] - cy;

    if (Math.abs(vy) > 100000) return [1, 0, 0, 0, 1, 0, 0, 0, 1];

    const hy = -1.0 / vy;

    const P_center = [1, 0, 0, 0, 1, 0, 0, hy, 1];
    const T1 = [1, 0, -cx, 0, 1, -cy, 0, 0, 1];
    const T2 = [1, 0, cx, 0, 1, cy, 0, 0, 1];

    return mul3(T2, mul3(P_center, T1));
}

// --- Robust Stats Helpers ---

function weightedQuantile(values, weights, q) {
    const n = values.length;
    if (n === 0) return 0;
    const items = new Array(n);
    let totalW = 0;
    for (let i = 0; i < n; i++) {
        const w = Number.isFinite(weights[i]) ? Math.max(0, weights[i]) : 0;
        items[i] = { v: values[i], w };
        totalW += w;
    }
    if (totalW <= 1e-12) {
        const sorted = [...values].sort((a, b) => a - b);
        return sorted[Math.floor((n - 1) * q)];
    }
    items.sort((a, b) => a.v - b.v);
    let acc = 0;
    const target = totalW * clamp(q, 0, 1);
    for (let i = 0; i < n; i++) {
        acc += items[i].w;
        if (acc >= target) return items[i].v;
    }
    return items[n - 1].v;
}

function robustVerticalLineStats(detectedLines, opts = {}) {
    const {
        toleranceDeg = 35,
        minLen = 0,
        minCount = 4,
        maxLines = 80,
    } = opts;

    const candidates = detectedLines
        .filter((l) => l.len >= minLen && Math.abs(l.angle - 90) < toleranceDeg)
        .sort((a, b) => b.len - a.len)
        .slice(0, maxLines);

    if (candidates.length < minCount) return null;

    const offsets = candidates.map((l) => l.angle - 90); // signed degrees
    const weights = candidates.map((l) => l.len);

    const med = weightedQuantile(offsets, weights, 0.5);
    const absDevFromMed = offsets.map((o) => Math.abs(o - med));
    const mad = weightedQuantile(absDevFromMed, weights, 0.5) || 1e-6;

    // Keep a robust inlier set; floor at ~1.5° to avoid over-pruning on very straight images.
    const inlierThr = Math.max(1.5, 3.0 * mad);
    const inliers = [];
    for (let i = 0; i < candidates.length; i++) {
        if (Math.abs(offsets[i] - med) <= inlierThr) inliers.push(candidates[i]);
    }
    if (inliers.length < minCount) return null;

    let wSum = 0;
    let meanOffset = 0;
    let meanAbs = 0;
    for (const l of inliers) {
        wSum += l.len;
        meanOffset += l.len * (l.angle - 90);
        meanAbs += l.len * Math.abs(l.angle - 90);
    }
    meanOffset /= wSum || 1;
    meanAbs /= wSum || 1;

    let varOffset = 0;
    for (const l of inliers) {
        const o = l.angle - 90;
        varOffset += l.len * (o - meanOffset) * (o - meanOffset);
    }
    varOffset /= wSum || 1;
    const stdOffset = Math.sqrt(varOffset);

    return {
        lines: inliers,
        count: inliers.length,
        meanOffset,
        meanAbs,
        stdOffset,
        medianOffset: med,
        madOffset: mad,
    };
}

function robustHorizontalLineStats(detectedLines, opts = {}) {
    const {
        toleranceDeg = 25,
        minLen = 0,
        minCount = 4,
        maxLines = 80,
    } = opts;

    // Convert [0, 180) angle to signed offset around 0 in (-90, 90]
    const signedOffset0 = (angleDeg) => {
        const a = ((angleDeg % 180) + 180) % 180; // [0,180)
        return ((a + 90) % 180) - 90; // [-90,90)
    };

    const candidates = detectedLines
        .map((l) => ({ ...l, hoff: signedOffset0(l.angle), habs: Math.abs(signedOffset0(l.angle)) }))
        .filter((l) => l.len >= minLen && l.habs < toleranceDeg)
        .sort((a, b) => b.len - a.len)
        .slice(0, maxLines);

    if (candidates.length < minCount) return null;

    const offsets = candidates.map((l) => l.hoff);
    const weights = candidates.map((l) => l.len);

    const med = weightedQuantile(offsets, weights, 0.5);
    const absDevFromMed = offsets.map((o) => Math.abs(o - med));
    const mad = weightedQuantile(absDevFromMed, weights, 0.5) || 1e-6;

    const inlierThr = Math.max(1.5, 3.0 * mad);
    const inliers = [];
    for (let i = 0; i < candidates.length; i++) {
        if (Math.abs(offsets[i] - med) <= inlierThr) inliers.push(candidates[i]);
    }
    if (inliers.length < minCount) return null;

    let wSum = 0;
    let meanOffset = 0;
    let meanAbs = 0;
    for (const l of inliers) {
        wSum += l.len;
        meanOffset += l.len * l.hoff;
        meanAbs += l.len * Math.abs(l.hoff);
    }
    meanOffset /= wSum || 1;
    meanAbs /= wSum || 1;

    let varOffset = 0;
    for (const l of inliers) {
        varOffset += l.len * (l.hoff - meanOffset) * (l.hoff - meanOffset);
    }
    varOffset /= wSum || 1;
    const stdOffset = Math.sqrt(varOffset);

    return {
        lines: inliers,
        count: inliers.length,
        meanOffset,
        meanAbs,
        stdOffset,
        medianOffset: med,
        madOffset: mad,
    };
}

function buildWarpBBox(H, w, h) {
    const pts = [
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ];
    const proj = pts.map((p) => {
        const r = appH(H, p[0], p[1]);
        return [r[0] / r[2], r[1] / r[2]];
    });
    const xs = proj.map((p) => p[0]);
    const ys = proj.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const bbW = maxX - minX;
    const bbH = maxY - minY;
    return { minX, minY, maxX, maxY, bbW, bbH };
}

function transformLineSegment(line, H) {
    const p1 = appH(H, line.x1, line.y1);
    const p2 = appH(H, line.x2, line.y2);
    const w1 = p1[2];
    const w2 = p2[2];
    if (!Number.isFinite(w1) || !Number.isFinite(w2) || Math.abs(w1) < 1e-8 || Math.abs(w2) < 1e-8) return null;
    const x1 = p1[0] / w1;
    const y1 = p1[1] / w1;
    const x2 = p2[0] / w2;
    const y2 = p2[1] / w2;
    if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) return null;
    const angle = ang180(Math.atan2(dy, dx));
    return { x1, y1, x2, y2, angle, len };
}

function evaluateCandidateCorrection(beforeVertStats, beforeHorizStats, Hcand, w, h) {
    if (!beforeVertStats || beforeVertStats.count < 4) return { shouldApply: false, reason: 'low_vertical_signal' };

    const { bbW, bbH } = buildWarpBBox(Hcand, w, h);
    if (!Number.isFinite(bbW) || !Number.isFinite(bbH) || bbW <= 1 || bbH <= 1) {
        return { shouldApply: false, reason: 'invalid_bbox' };
    }

    // Keep evaluation bounded: large bboxes usually indicate unstable H (and full-res will be rejected anyway).
    const areaScale = (bbW * bbH) / (w * h);
    if (areaScale > 3.0 || bbW < w * 0.55) {
        return { shouldApply: false, reason: 'unstable_bbox' };
    }

    // Evaluate on the SAME inlier line set by projecting endpoints through Hcand.
    // This avoids false negatives caused by re-running Hough after a warp (which can introduce many noisy edges).
    const projectedVert = [];
    for (const l of beforeVertStats.lines) {
        const pl = transformLineSegment(l, Hcand);
        if (pl) projectedVert.push(pl);
    }
    if (projectedVert.length < 4) return { shouldApply: false, reason: 'eval_project_fail' };

    const afterVertStats = robustVerticalLineStats(projectedVert, {
        toleranceDeg: 35,
        minLen: Math.max(w, h) * 0.08,
        minCount: 4,
        maxLines: 120,
    });
    if (!afterVertStats) return { shouldApply: false, reason: 'eval_no_lines' };

    const improveAbs = beforeVertStats.meanAbs - afterVertStats.meanAbs;
    const improveRel = improveAbs / Math.max(0.001, beforeVertStats.meanAbs);

    // If the image is already quite straight, require stronger evidence to apply any warp.
    const alreadyGood = beforeVertStats.meanAbs < 1.8 && beforeVertStats.stdOffset < 2.2;
    const passImprove = improveAbs >= (alreadyGood ? 0.35 : 0.25) || improveRel >= (alreadyGood ? 0.18 : 0.12);

    // Avoid "fixing" into a more chaotic set of verticals.
    const passStability = afterVertStats.stdOffset <= beforeVertStats.stdOffset + 0.9;

    // Horizontal guardrail: a good vertical-upright warp should not noticeably tilt horizontals.
    // Only enforce when we have enough horizontal evidence in the original.
    let passHorizontal = true;
    let horizMetrics = null;
    if (beforeHorizStats && beforeHorizStats.count >= 4) {
        const projectedHoriz = [];
        for (const l of beforeHorizStats.lines) {
            const pl = transformLineSegment(l, Hcand);
            if (pl) projectedHoriz.push(pl);
        }
        if (projectedHoriz.length >= 4) {
            const afterHorizStats = robustHorizontalLineStats(projectedHoriz, {
                toleranceDeg: 25,
                minLen: Math.max(w, h) * 0.08,
                minCount: 4,
                maxLines: 120,
            });
            if (afterHorizStats) {
                const worsenAbs = afterHorizStats.meanAbs - beforeHorizStats.meanAbs;
                const worsenRel = worsenAbs / Math.max(0.001, beforeHorizStats.meanAbs);
                const allowAbs = Math.max(0.55, beforeHorizStats.meanAbs * 0.25);
                const passWorsen = worsenAbs <= allowAbs || worsenRel <= 0.25;
                const passHorizStd = afterHorizStats.stdOffset <= beforeHorizStats.stdOffset + 1.2;
                const veryGoodBefore = beforeHorizStats.meanAbs < 1.2 && beforeHorizStats.stdOffset < 1.8;
                const passVeryGood = !veryGoodBefore || afterHorizStats.meanAbs <= 1.7;

                passHorizontal = passWorsen && passHorizStd && passVeryGood;
                horizMetrics = {
                    before: { meanAbs: beforeHorizStats.meanAbs, std: beforeHorizStats.stdOffset, count: beforeHorizStats.count },
                    after: { meanAbs: afterHorizStats.meanAbs, std: afterHorizStats.stdOffset, count: afterHorizStats.count },
                    worsenAbs,
                    worsenRel,
                };
            }
        }
    }

    return {
        shouldApply: passImprove && passStability && passHorizontal,
        reason: !passImprove
            ? 'eval_no_improve'
            : !passStability
                ? 'eval_unstable'
                : !passHorizontal
                    ? 'eval_horizon_tilt'
                    : 'ok',
        metrics: {
            before: { meanAbs: beforeVertStats.meanAbs, std: beforeVertStats.stdOffset, count: beforeVertStats.count },
            after: { meanAbs: afterVertStats.meanAbs, std: afterVertStats.stdOffset, count: afterVertStats.count },
            improveAbs,
            improveRel,
            areaScale,
            horizontals: horizMetrics,
        },
    };
}

function scoreEvalResult(evalRes) {
    if (!evalRes || !evalRes.shouldApply || !evalRes.metrics) return Infinity;
    const m = evalRes.metrics;
    const v = m.after;
    const h = m.horizontals?.after;
    const horizPenalty = h ? 0.35 * h.meanAbs + 0.08 * h.std : 0;
    return v.meanAbs + 0.12 * v.std + horizPenalty + 0.05 * (m.areaScale || 1);
}

function buildCandidateFromVp(vp, rotateDeg, aw, ah) {
    const R = rotC(rotateDeg, aw, ah);
    const vpRot = appH(R, vp.x, vp.y);
    const vpx_new = vpRot[0] / vpRot[2];
    const vpy_new = vpRot[1] / vpRot[2];

    let H_persp = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    if (Math.abs(vpy_new) > ah * 1.5) {
        H_persp = onePointVerticalH([vpx_new, vpy_new], aw, ah);
    }
    const H = mul3(H_persp, R);
    const perspStrength = Math.abs(H[7]) * ah;
    return { H, rotateDeg, perspStrength };
}

// --- RANSAC Helpers ---

function makeRng(seed) {
    let t = seed >>> 0;
    return () => {
        t += 0x6d2b79f5;
        let x = t;
        x = Math.imul(x ^ (x >>> 15), 1 | x);
        x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
        return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
}

function ransacVP(lines, rng, width, height) {
    if (lines.length < 2) return null;
    const n = lines.length;
    const iters = Math.min(500, n * n);
    let best = { x: 0, y: 0, count: 0, score: 0 };

    for (let k = 0; k < iters; k++) {
        const i = (rng() * n) | 0;
        let j = (rng() * n) | 0;
        if (i === j) j = (j + 1) % n;

        const p = inter(lines[i], lines[j]);
        if (!p) continue;
        const [px, py] = p;

        if (!Number.isFinite(px) || !Number.isFinite(py)) continue;

        let count = 0;
        let sumDist = 0;
        const threshold = Math.max(width, height) * 0.02;

        for (let m = 0; m < n; m++) {
            const d = dist(lines[m], px, py);
            if (d < threshold) {
                count++;
                sumDist += d;
            }
        }

        const isInside = (px > 0 && px < width && py > 0 && py < height);
        if (isInside) continue;

        const score = count * 1000 - sumDist;

        if (score > best.score) {
            best = { x: px, y: py, count, score };
        }
    }

    if (best.count < Math.max(4, n * 0.15)) return null;
    return best;
}

/**
 * CORE ALGORITHM: Largest Inscribed Rectangle in Histogram
 */
function autoCropToLargestInscribedRect(dst, maskDst, cv) {
    const rows = maskDst.rows;
    const cols = maskDst.cols;
    const data = maskDst.data; // Uint8Array

    // Step 1: Build height histogram
    const heights = new Uint16Array(rows * cols);

    for (let c = 0; c < cols; c++) {
        heights[c] = data[c] > 127 ? 1 : 0;
    }
    for (let r = 1; r < rows; r++) {
        const rowOffset = r * cols;
        const prevRowOffset = (r - 1) * cols;
        for (let c = 0; c < cols; c++) {
            if (data[rowOffset + c] > 127) {
                heights[rowOffset + c] = heights[prevRowOffset + c] + 1;
            } else {
                heights[rowOffset + c] = 0;
            }
        }
    }

    // Step 2: For each row, use Monotonic Stack to find max rect
    let bestArea = 0;
    let bestRect = { x: 0, y: 0, w: 0, h: 0 };

    const largestRectInHistogram = (row) => {
        const stack = [];
        let maxArea = 0;
        let maxRect = null;
        const rowOffset = row * cols;

        for (let i = 0; i <= cols; i++) {
            const h = (i === cols) ? 0 : heights[rowOffset + i];
            let startIdx = i;

            while (stack.length > 0 && stack[stack.length - 1].h > h) {
                const top = stack.pop();
                const width = i - top.idx;
                const area = top.h * width;

                if (area > maxArea) {
                    maxArea = area;
                    maxRect = {
                        x: top.idx,
                        y: row - top.h + 1,
                        w: width,
                        h: top.h
                    };
                }
                startIdx = top.idx;
            }
            stack.push({ idx: startIdx, h: h });
        }
        return { area: maxArea, rect: maxRect };
    };

    for (let r = 0; r < rows; r++) {
        const result = largestRectInHistogram(r);
        if (result.area > bestArea && result.rect) {
            bestArea = result.area;
            bestRect = result.rect;
        }
    }

    // Step 3: Crop
    if (bestArea > 100 && bestRect.w > 10 && bestRect.h > 10) {
        try {
            const rect = new cv.Rect(bestRect.x, bestRect.y, bestRect.w, bestRect.h);
            const roi = dst.roi(rect);
            const cropped = roi.clone();
            roi.delete();
            return cropped;
        } catch (e) {
            console.warn("Crop ROI failed", e);
            return null;
        }
    }
    return null;
}


// --- Main Exports ---

/**
 * Automatically crops black or transparent borders from an image.
 */
export const autoCrop = async (imageSrc) => {
    return new Promise((resolve) => {
        const cv = window.cv;
        if (!cv || !cv.Mat) {
            resolve(imageSrc);
            return;
        }

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = imageSrc;

        img.onload = () => {
            try {
                const src = cv.imread(img);
                let mask = new cv.Mat();
                if (src.channels() === 4) {
                    const channels = new cv.MatVector();
                    cv.split(src, channels);
                    const alpha = channels.get(3);
                    cv.threshold(alpha, mask, 10, 255, cv.THRESH_BINARY);
                    alpha.delete();
                    channels.delete();
                } else {
                    const gray = new cv.Mat();
                    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
                    cv.threshold(gray, mask, 10, 255, cv.THRESH_BINARY);
                    gray.delete();
                }

                const croppedMat = autoCropToLargestInscribedRect(src, mask, cv);

                let result = imageSrc;
                if (croppedMat) {
                    const canvas = document.createElement('canvas');
                    cv.imshow(canvas, croppedMat);
                    result = canvas.toDataURL('image/jpeg', 0.95);
                    croppedMat.delete();
                }

                src.delete();
                mask.delete();
                resolve(result);

            } catch (e) {
                console.error("AutoCrop failed", e);
                resolve(imageSrc);
            }
        };
        img.onerror = () => resolve(imageSrc);
    });
};

/**
 * Automatically corrects the perspective of an image (Auto Upright).
 * Detects vertical lines and warps the image to make them parallel.
 */
export const autoUpright = async (imageSrc, options = {}) => {
    const { autoCrop = true } = options;

    return new Promise((resolve) => {
        const cv = window.cv;
        if (!cv || !cv.Mat) {
            resolve({ success: false, error: "OpenCV not loaded" });
            return;
        }

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = imageSrc;

        img.onload = async () => {
            let src = null,
                gray = null,
                blur = null,
                edges = null,
                lines = null,
                srcFull = null,
                dst = null,
                maskSrc = null,
                maskDst = null;

            try {
                const W = img.width;
                const H = img.height;
                let chosenEvalMetrics = null;

                // 1. Downscale for Analysis
                const analysisScale = Math.min(1, 1000 / Math.max(W, H));
                const aw = Math.round(W * analysisScale);
                const ah = Math.round(H * analysisScale);

                const canvas = document.createElement('canvas');
                canvas.width = aw;
                canvas.height = ah;
                const ctx = canvas.getContext('2d', { willReadFrequently: true });
                if (!ctx) throw new Error("Context failed");
                ctx.drawImage(img, 0, 0, aw, ah);

                // 2. Detect Lines
                src = cv.imread(canvas);
                gray = new cv.Mat();
                cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
                blur = new cv.Mat();
                cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 1.5, 1.5, cv.BORDER_DEFAULT);

                edges = new cv.Mat();
                cv.Canny(blur, edges, 50, 150);

                lines = new cv.Mat();
                const minLineLen = Math.max(aw, ah) * 0.08;
                const maxLineGap = Math.max(aw, ah) * 0.03;
                cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, minLineLen, maxLineGap);

                // 3. Process Lines
                const detectedLines = [];
                for (let i = 0; i < lines.rows; i++) {
                    const x1 = lines.data32S[i * 4];
                    const y1 = lines.data32S[i * 4 + 1];
                    const x2 = lines.data32S[i * 4 + 2];
                    const y2 = lines.data32S[i * 4 + 3];
                    const dx = x2 - x1;
                    const dy = y2 - y1;
                    const len = Math.hypot(dx, dy);
                    const angle = ang180(Math.atan2(dy, dx));
                    detectedLines.push({ x1, y1, x2, y2, angle, len });
                }

                // 4. Robust vertical signal extraction
                const statsVertBefore = robustVerticalLineStats(detectedLines, {
                    toleranceDeg: 35,
                    minLen: minLineLen * 1.1,
                    minCount: 4,
                });

                if (!statsVertBefore) {
                    console.warn("Not enough vertical lines found. Skipping correction.");
                    resolve({ success: true, image: imageSrc });
                    return;
                }

                const statsHorizBefore = robustHorizontalLineStats(detectedLines, {
                    toleranceDeg: 25,
                    minLen: minLineLen * 1.1,
                    minCount: 4,
                });

                // Fast-path: already very straight and consistent => skip.
                // (Use robust stats so a few noisy segments don't trigger a bad warp.)
                const SKIP_THRESHOLD_MEAN_ABS = 1.2; // degrees
                const SKIP_THRESHOLD_STD = 1.8; // degrees
                if (statsVertBefore.meanAbs < SKIP_THRESHOLD_MEAN_ABS && statsVertBefore.stdOffset < SKIP_THRESHOLD_STD) {
                    console.log(
                        `Image already straight. meanAbs=${statsVertBefore.meanAbs.toFixed(2)}°, std=${statsVertBefore.stdOffset.toFixed(2)}°. Skipping.`,
                    );
                    resolve({ success: true, image: imageSrc, skipped: true, reason: 'already_straight' });
                    return;
                }

                const lineEqs = statsVertBefore.lines.map(l => {
                    return normL(line2(l.x1, l.y1, l.x2, l.y2));
                });

                // 5. RANSAC for Vanishing Point
                const rng = makeRng(W * H + statsVertBefore.lines.length);
                const vp = ransacVP(lineEqs, rng, aw, ah);

                let FinalH = [1, 0, 0, 0, 1, 0, 0, 0, 1];
                let totalRotation = 0;  // Track total rotation for skip check
                let perspStrength = 0;

                if (vp) {
                    // Build multiple candidates; choose the best by self-check.
                    const cx = aw / 2;
                    const cy = ah / 2;
                    const vx = vp.x - cx;
                    const vy = vp.y - cy;
                    const vpAngle = (Math.atan2(vy, vx) * 180) / Math.PI;

                    // Previous "VP direction" rotation (worked well before).
                    let rotateVp = vy < 0 ? -90 - vpAngle : 90 - vpAngle;
                    rotateVp = clamp(rotateVp, -15, 15);

                    // Robust "roll from vertical lines" rotation (helps when VP is noisy).
                    const rotateRoll = clamp(-statsVertBefore.meanOffset, -15, 15);

                    const candidates = [
                        buildCandidateFromVp(vp, rotateVp, aw, ah),
                        buildCandidateFromVp(vp, rotateRoll, aw, ah),
                    ];

                    // Skip if all candidates are negligible.
                    const nonNegligible = candidates.filter(
                        (c) => Math.abs(c.rotateDeg) >= 0.7 || c.perspStrength >= 0.002,
                    );
                    if (nonNegligible.length === 0) {
                        resolve({ success: true, image: imageSrc, skipped: true, reason: 'correction_negligible' });
                        return;
                    }

                    const evals = nonNegligible.map((c) => ({
                        cand: c,
                        eval: evaluateCandidateCorrection(statsVertBefore, statsHorizBefore, c.H, aw, ah),
                    }));
                    evals.sort((a, b) => {
                        const sa = scoreEvalResult(a.eval);
                        const sb = scoreEvalResult(b.eval);
                        if (!Number.isFinite(sa) && !Number.isFinite(sb)) return 0;
                        if (!Number.isFinite(sa)) return 1;
                        if (!Number.isFinite(sb)) return -1;
                        return sa - sb;
                    });

                    const best = evals[0];
                    if (!best || !best.eval.shouldApply) {
                        const fallback = best?.eval || evals.find((e) => e.eval.shouldApply)?.eval;
                        resolve({
                            success: true,
                            image: imageSrc,
                            skipped: true,
                            reason: `gate_${(fallback && fallback.reason) ? fallback.reason : 'eval_no_improve'}`,
                            metrics: fallback?.metrics,
                        });
                        return;
                    }

                    FinalH = best.cand.H;
                    totalRotation = Math.abs(best.cand.rotateDeg);
                    perspStrength = best.cand.perspStrength;

                    // Preserve metrics for demo/debug
                    chosenEvalMetrics = best.eval.metrics;
                } else {
                    const rotNeeded = -statsVertBefore.meanOffset;
                    totalRotation = Math.abs(rotNeeded);
                    if (Math.abs(rotNeeded) < 10) {
                        FinalH = rotC(rotNeeded, aw, ah);
                    }
                }

                // For non-VP case, keep the single-candidate self-check.
                if (!vp) {
                    if (totalRotation < 0.7 && perspStrength < 0.002) {
                        resolve({ success: true, image: imageSrc, skipped: true, reason: 'correction_negligible' });
                        return;
                    }
                    const evalRes = evaluateCandidateCorrection(statsVertBefore, statsHorizBefore, FinalH, aw, ah);
                    if (!evalRes.shouldApply) {
                        resolve({
                            success: true,
                            image: imageSrc,
                            skipped: true,
                            reason: `gate_${evalRes.reason}`,
                            metrics: evalRes.metrics,
                        });
                        return;
                    }
                    chosenEvalMetrics = evalRes.metrics;
                }

                // 6. Bounding Box & Warping
                const Sdown = [analysisScale, 0, 0, 0, analysisScale, 0, 0, 0, 1];
                const Sup = [1 / analysisScale, 0, 0, 0, 1 / analysisScale, 0, 0, 0, 1];
                const FullH = mul3(Sup, mul3(FinalH, Sdown));

                const pts = [[0, 0], [W, 0], [W, H], [0, H]];
                const projPts = pts.map(p => {
                    const res = appH(FullH, p[0], p[1]);
                    return [res[0] / res[2], res[1] / res[2]];
                });

                const xs = projPts.map(p => p[0]);
                const ys = projPts.map(p => p[1]);
                const bbW = Math.max(...xs) - Math.min(...xs);
                const bbH = Math.max(...ys) - Math.min(...ys);

                // Safety: Avoid infinite warping
                if (bbW * bbH > (W * H * 3.0) || bbW < W * 0.5) {
                    resolve({ success: true, image: imageSrc });
                    return;
                }

                const minX = Math.min(...xs);
                const minY = Math.min(...ys);

                const T = [1, 0, -minX, 0, 1, -minY, 0, 0, 1];
                const FinalMatrix = mul3(T, FullH);

                const finalW = Math.round(bbW);
                const finalH = Math.round(bbH);

                srcFull = cv.imread(img);
                dst = new cv.Mat();
                const M = cv.matFromArray(3, 3, cv.CV_64F, FinalMatrix);

                // Step A: Warp the Image
                // Use transparent border to easily detect edges later
                cv.warpPerspective(srcFull, dst, M, new cv.Size(finalW, finalH), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));

                if (autoCrop) {
                    // Step B: Create and Warp the Mask
                    maskSrc = new cv.Mat(srcFull.rows, srcFull.cols, cv.CV_8UC1, new cv.Scalar(255));
                    maskDst = new cv.Mat();
                    cv.warpPerspective(maskSrc, maskDst, M, new cv.Size(finalW, finalH), cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));

                    // Step C: Run Largest Inscribed Rect Algorithm
                    const cropped = autoCropToLargestInscribedRect(dst, maskDst, cv);
                    if (cropped) {
                        dst.delete();
                        dst = cropped;
                    }
                }

                const tempCanvas = document.createElement('canvas');
                cv.imshow(tempCanvas, dst);
                resolve({
                    success: true,
                    image: tempCanvas.toDataURL('image/jpeg', 0.95),
                    metrics: chosenEvalMetrics || undefined,
                });

                M.delete();
                srcFull.delete();
                dst.delete();
                if (maskSrc) maskSrc.delete();
                if (maskDst) maskDst.delete();

            } catch (e) {
                console.error("OpenCV Error", e);
                resolve({ success: false, error: String(e) });
            } finally {
                if (src) src.delete();
                if (gray) gray.delete();
                if (blur) blur.delete();
                if (edges) edges.delete();
                if (lines) lines.delete();
            }
        };
        img.onerror = () => resolve({ success: false, error: "Image load failed" });
    });
};

// Also expose globally for non-module usage (e.g. direct file:// access)
if (typeof window !== 'undefined') {
    window.autoCrop = autoCrop;
    window.autoUpright = autoUpright;
}
