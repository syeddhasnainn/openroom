use wasm_bindgen::prelude::*;

// ============ TONE CURVE (Optimized with LUT) ============

/// Monotonic cubic spline interpolation for smooth curves
/// Takes control points (x, y pairs flattened) and generates a 256-entry LUT
fn cubic_spline_lut(points: &[(f32, f32)]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    
    if points.len() < 2 {
        // Identity curve
        for i in 0..256 {
            lut[i] = i as u8;
        }
        return lut;
    }
    
    let n = points.len();
    
    // For each output value, find the corresponding input
    for i in 0..256 {
        let x = i as f32 / 255.0;
        
        // Find which segment we're in
        let mut seg = 0;
        for j in 0..n - 1 {
            if x >= points[j].0 && x <= points[j + 1].0 {
                seg = j;
                break;
            }
            if j == n - 2 {
                seg = j;
            }
        }
        
        // Handle edge cases
        if x <= points[0].0 {
            lut[i] = (points[0].1 * 255.0).clamp(0.0, 255.0) as u8;
            continue;
        }
        if x >= points[n - 1].0 {
            lut[i] = (points[n - 1].1 * 255.0).clamp(0.0, 255.0) as u8;
            continue;
        }
        
        let x0 = points[seg].0;
        let x1 = points[seg + 1].0;
        let y0 = points[seg].1;
        let y1 = points[seg + 1].1;
        
        // Catmull-Rom spline for smooth interpolation
        let t = if (x1 - x0).abs() > 0.0001 { (x - x0) / (x1 - x0) } else { 0.0 };
        
        // Get tangents using neighboring points
        let m0 = if seg > 0 {
            let y_prev = points[seg - 1].1;
            let x_prev = points[seg - 1].0;
            let dx = x1 - x_prev;
            if dx.abs() > 0.0001 { (y1 - y_prev) / dx * (x1 - x0) } else { 0.0 }
        } else {
            y1 - y0
        };
        
        let m1 = if seg < n - 2 {
            let y_next = points[seg + 2].1;
            let x_next = points[seg + 2].0;
            let dx = x_next - x0;
            if dx.abs() > 0.0001 { (y_next - y0) / dx * (x1 - x0) } else { 0.0 }
        } else {
            y1 - y0
        };
        
        // Hermite basis functions
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        
        let y = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1;
        lut[i] = (y * 255.0).clamp(0.0, 255.0) as u8;
    }
    
    lut
}

/// Apply tone curve using pre-computed LUTs for maximum speed
/// points_rgb, points_r, points_g, points_b are flattened arrays of (x, y) control points
/// Each array should have an even number of elements (x0, y0, x1, y1, ...)
/// Coordinates are in 0-1 range
#[wasm_bindgen]
pub fn apply_tone_curve(
    data: &[u8],
    points_rgb: &[f32],
    points_r: &[f32],
    points_g: &[f32],
    points_b: &[f32],
) -> Vec<u8> {
    // Parse control points
    let parse_points = |flat: &[f32]| -> Vec<(f32, f32)> {
        flat.chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| (c[0], c[1]))
            .collect()
    };
    
    let rgb_points = parse_points(points_rgb);
    let r_points = parse_points(points_r);
    let g_points = parse_points(points_g);
    let b_points = parse_points(points_b);
    
    // Check if we have any actual curves (not just identity)
    let is_identity = |pts: &[(f32, f32)]| -> bool {
        if pts.len() == 2 {
            let p0 = pts[0];
            let p1 = pts[1];
            (p0.0 - 0.0).abs() < 0.001 && (p0.1 - 0.0).abs() < 0.001 &&
            (p1.0 - 1.0).abs() < 0.001 && (p1.1 - 1.0).abs() < 0.001
        } else {
            false
        }
    };
    
    let rgb_identity = rgb_points.is_empty() || is_identity(&rgb_points);
    let r_identity = r_points.is_empty() || is_identity(&r_points);
    let g_identity = g_points.is_empty() || is_identity(&g_points);
    let b_identity = b_points.is_empty() || is_identity(&b_points);
    
    // Early return if all curves are identity
    if rgb_identity && r_identity && g_identity && b_identity {
        return data.to_vec();
    }
    
    // Build LUTs (this is O(256) per curve, done once)
    let rgb_lut = if !rgb_identity { cubic_spline_lut(&rgb_points) } else { 
        let mut lut = [0u8; 256];
        for i in 0..256 { lut[i] = i as u8; }
        lut
    };
    
    let r_lut = if !r_identity { cubic_spline_lut(&r_points) } else { 
        let mut lut = [0u8; 256];
        for i in 0..256 { lut[i] = i as u8; }
        lut
    };
    
    let g_lut = if !g_identity { cubic_spline_lut(&g_points) } else { 
        let mut lut = [0u8; 256];
        for i in 0..256 { lut[i] = i as u8; }
        lut
    };
    
    let b_lut = if !b_identity { cubic_spline_lut(&b_points) } else { 
        let mut lut = [0u8; 256];
        for i in 0..256 { lut[i] = i as u8; }
        lut
    };
    
    // Compose LUTs: first apply RGB curve, then individual channel curves
    // This creates combined LUTs for each channel (O(256) per channel)
    let mut final_r_lut = [0u8; 256];
    let mut final_g_lut = [0u8; 256];
    let mut final_b_lut = [0u8; 256];
    
    for i in 0..256 {
        let rgb_val = rgb_lut[i] as usize;
        final_r_lut[i] = r_lut[rgb_val];
        final_g_lut[i] = g_lut[rgb_val];
        final_b_lut[i] = b_lut[rgb_val];
    }
    
    // Apply combined LUTs in a single pass - O(1) per pixel, O(n) total
    let mut result = data.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        chunk[0] = final_r_lut[chunk[0] as usize];
        chunk[1] = final_g_lut[chunk[1] as usize];
        chunk[2] = final_b_lut[chunk[2] as usize];
        // Alpha channel unchanged
    }
    
    result
}

// ============ FILM GRAIN ============

/// Fast pseudo-random number generator (xorshift)
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Apply film grain effect
/// amount: 0-100 (intensity of grain)
/// size: 0-100 (grain size, 0 = fine, 100 = coarse)
/// roughness: 0-100 (how varied the grain is)
#[wasm_bindgen]
pub fn apply_grain(
    data: &[u8],
    width: u32,
    height: u32,
    amount: f32,
    size: f32,
    roughness: f32,
) -> Vec<u8> {
    if amount <= 0.0 {
        return data.to_vec();
    }
    
    let mut result = data.to_vec();
    let w = width as usize;
    let h = height as usize;
    
    let intensity = amount / 100.0 * 0.5; // Max 50% intensity
    let grain_size = 1.0 + (size / 100.0) * 3.0; // 1-4 pixel grain
    let rough = 0.5 + (roughness / 100.0) * 0.5; // 0.5-1.0 roughness
    
    // Use image dimensions as seed for reproducible grain
    let mut rng_state: u32 = (width.wrapping_mul(height)).wrapping_add(12345);
    
    // Pre-generate grain pattern for efficiency
    let grain_w = ((w as f32 / grain_size).ceil() as usize).max(1);
    let grain_h = ((h as f32 / grain_size).ceil() as usize).max(1);
    let mut grain_map: Vec<f32> = Vec::with_capacity(grain_w * grain_h);
    
    for _ in 0..(grain_w * grain_h) {
        let r = (xorshift32(&mut rng_state) as f32) / (u32::MAX as f32);
        // Map to -1 to 1 range with roughness affecting distribution
        let grain_val = ((r - 0.5) * 2.0).powf(rough.recip());
        grain_map.push(grain_val);
    }
    
    // Apply grain with bilinear sampling for smoother large grains
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            
            // Sample grain with size scaling
            let gx = (x as f32 / grain_size).min((grain_w - 1) as f32);
            let gy = (y as f32 / grain_size).min((grain_h - 1) as f32);
            
            let gx0 = gx.floor() as usize;
            let gy0 = gy.floor() as usize;
            let gx1 = (gx0 + 1).min(grain_w - 1);
            let gy1 = (gy0 + 1).min(grain_h - 1);
            
            let fx = gx - gx0 as f32;
            let fy = gy - gy0 as f32;
            
            // Bilinear interpolation for smooth grain at larger sizes
            let g00 = grain_map[gy0 * grain_w + gx0];
            let g10 = grain_map[gy0 * grain_w + gx1];
            let g01 = grain_map[gy1 * grain_w + gx0];
            let g11 = grain_map[gy1 * grain_w + gx1];
            
            let grain = g00 * (1.0 - fx) * (1.0 - fy)
                      + g10 * fx * (1.0 - fy)
                      + g01 * (1.0 - fx) * fy
                      + g11 * fx * fy;
            
            // Get luminance for intensity modulation (grain more visible in midtones)
            let lum = (result[idx] as f32 * 0.299 
                     + result[idx + 1] as f32 * 0.587 
                     + result[idx + 2] as f32 * 0.114) / 255.0;
            
            // Reduce grain in shadows and highlights
            let lum_mask = 4.0 * lum * (1.0 - lum); // Peaks at 0.5
            let final_grain = grain * intensity * 255.0 * (0.3 + 0.7 * lum_mask);
            
            // Apply to all channels equally for monochromatic grain
            for c in 0..3 {
                let val = result[idx + c] as f32 + final_grain;
                result[idx + c] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    result
}

// ============ HSL COLOR ADJUSTMENTS ============

/// Convert RGB to HSL
fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    
    if max == min {
        return (0.0, 0.0, l);
    }
    
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    
    let h = if max == r {
        let h = (g - b) / d;
        if g < b { h + 6.0 } else { h }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    
    (h / 6.0, s, l) // h is now 0-1
}

/// Convert HSL to RGB
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    
    fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
        if t < 0.0 { t += 1.0; }
        if t > 1.0 { t -= 1.0; }
        if t < 1.0/6.0 { return p + (q - p) * 6.0 * t; }
        if t < 1.0/2.0 { return q; }
        if t < 2.0/3.0 { return p + (q - p) * (2.0/3.0 - t) * 6.0; }
        p
    }
    
    (
        hue_to_rgb(p, q, h + 1.0/3.0),
        hue_to_rgb(p, q, h),
        hue_to_rgb(p, q, h - 1.0/3.0),
    )
}

/// Get the weight for how much a hue belongs to each color channel
/// Returns weights for: Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta
fn get_color_weights(hue: f32) -> [f32; 8] {
    // Hue ranges (in 0-1 scale):
    // Red: 0.000 (0°) / 1.000 (360°)
    // Orange: 0.083 (30°)
    // Yellow: 0.167 (60°)
    // Green: 0.333 (120°)
    // Aqua: 0.500 (180°)
    // Blue: 0.667 (240°)
    // Purple: 0.750 (270°)
    // Magenta: 0.833 (300°)
    
    let centers = [0.0, 0.083, 0.167, 0.333, 0.500, 0.667, 0.750, 0.833];
    let mut weights = [0.0f32; 8];
    
    for (i, &center) in centers.iter().enumerate() {
        let mut dist = (hue - center).abs();
        // Handle wrap-around for red
        if i == 0 {
            dist = dist.min((hue - 1.0).abs()).min((1.0 - hue).min(hue));
        }
        
        // Use a smooth falloff (cosine-based) with ~60° range
        let range = 0.083; // ~30° half-width
        if dist < range {
            weights[i] = ((1.0 - dist / range) * std::f32::consts::PI / 2.0).cos().powi(2);
        }
    }
    
    // Normalize weights
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    
    weights
}

/// HSL adjustment parameters for all 8 color channels
/// Each channel has: hue shift (-180 to 180), saturation (-100 to 100), luminance (-100 to 100)
#[wasm_bindgen]
pub fn adjust_hsl(
    data: &[u8],
    // Red
    red_hue: f32, red_sat: f32, red_lum: f32,
    // Orange
    orange_hue: f32, orange_sat: f32, orange_lum: f32,
    // Yellow
    yellow_hue: f32, yellow_sat: f32, yellow_lum: f32,
    // Green
    green_hue: f32, green_sat: f32, green_lum: f32,
    // Aqua
    aqua_hue: f32, aqua_sat: f32, aqua_lum: f32,
    // Blue
    blue_hue: f32, blue_sat: f32, blue_lum: f32,
    // Purple
    purple_hue: f32, purple_sat: f32, purple_lum: f32,
    // Magenta
    magenta_hue: f32, magenta_sat: f32, magenta_lum: f32,
) -> Vec<u8> {
    let hue_shifts = [
        red_hue / 360.0, orange_hue / 360.0, yellow_hue / 360.0, green_hue / 360.0,
        aqua_hue / 360.0, blue_hue / 360.0, purple_hue / 360.0, magenta_hue / 360.0,
    ];
    let sat_shifts = [
        red_sat / 100.0, orange_sat / 100.0, yellow_sat / 100.0, green_sat / 100.0,
        aqua_sat / 100.0, blue_sat / 100.0, purple_sat / 100.0, magenta_sat / 100.0,
    ];
    let lum_shifts = [
        red_lum / 100.0, orange_lum / 100.0, yellow_lum / 100.0, green_lum / 100.0,
        aqua_lum / 100.0, blue_lum / 100.0, purple_lum / 100.0, magenta_lum / 100.0,
    ];
    
    let mut result = data.to_vec();
    
    for chunk in result.chunks_exact_mut(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        
        let (h, s, l) = rgb_to_hsl(r, g, b);
        
        // Skip near-grayscale pixels
        if s < 0.05 {
            continue;
        }
        
        let weights = get_color_weights(h);
        
        // Calculate weighted adjustments
        let mut hue_adj = 0.0f32;
        let mut sat_adj = 0.0f32;
        let mut lum_adj = 0.0f32;
        
        for i in 0..8 {
            hue_adj += weights[i] * hue_shifts[i];
            sat_adj += weights[i] * sat_shifts[i];
            lum_adj += weights[i] * lum_shifts[i];
        }
        
        // Apply adjustments
        let mut new_h = h + hue_adj;
        if new_h < 0.0 { new_h += 1.0; }
        if new_h > 1.0 { new_h -= 1.0; }
        
        let new_s = (s + sat_adj * s).clamp(0.0, 1.0);
        let new_l = (l + lum_adj * 0.5).clamp(0.0, 1.0);
        
        let (new_r, new_g, new_b) = hsl_to_rgb(new_h, new_s, new_l);
        
        chunk[0] = (new_r * 255.0).clamp(0.0, 255.0) as u8;
        chunk[1] = (new_g * 255.0).clamp(0.0, 255.0) as u8;
        chunk[2] = (new_b * 255.0).clamp(0.0, 255.0) as u8;
    }
    
    result
}

#[wasm_bindgen]
pub fn grayscale(data: &[u8]) -> Vec<u8> {
    let mut result = data.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        let avg = ((chunk[0] as u32 + chunk[1] as u32 + chunk[2] as u32) / 3) as u8;
        chunk[0] = avg;
        chunk[1] = avg;
        chunk[2] = avg;
    }
    result
}

/// Build a 256-entry lookup table for fast per-pixel processing
fn build_lut(
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
) -> [u8; 256] {
    let mut lut = [0u8; 256];
    
    let exposure_mult = 2.0_f32.powf(exposure);
    let contrast_factor = (contrast + 100.0) / 100.0;
    let highlights_amt = highlights / 100.0;
    let shadows_amt = shadows / 100.0;
    let whites_amt = whites / 100.0;
    let blacks_amt = blacks / 100.0;

    for i in 0..256 {
        let mut v = i as f32 / 255.0;
        
        v *= exposure_mult;
        v = (v - 0.5) * contrast_factor + 0.5;
        
        if v < 0.25 {
            let w = 1.0 - (v / 0.25);
            v += blacks_amt * 0.3 * w * w;
        }
        if v < 0.5 {
            let w = if v < 0.1 { v / 0.1 } else { 1.0 - ((v - 0.1) / 0.4) };
            v += shadows_amt * 0.4 * w.max(0.0);
        }
        if v > 0.5 {
            let w = if v > 0.9 { 1.0 - ((v - 0.9) / 0.1) } else { (v - 0.5) / 0.4 };
            v -= highlights_amt * 0.4 * w.max(0.0);
        }
        if v > 0.75 {
            let w = (v - 0.75) / 0.25;
            v += whites_amt * 0.3 * w * w;
        }
        
        lut[i] = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }
    
    lut
}

#[wasm_bindgen]
pub fn adjust_image(
    data: &[u8],
    exposure: f32,
    contrast: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
) -> Vec<u8> {
    let lut = build_lut(exposure, contrast, highlights, shadows, whites, blacks);
    
    let mut result = data.to_vec();
    for chunk in result.chunks_exact_mut(4) {
        chunk[0] = lut[chunk[0] as usize];
        chunk[1] = lut[chunk[1] as usize];
        chunk[2] = lut[chunk[2] as usize];
    }
    
    result
}

// ============ FILM PRESETS ============

/// Film preset parameters for per-channel color grading
struct FilmProfile {
    // Per-channel adjustments (shadows, midtones, highlights)
    r_shadows: f32,
    r_midtones: f32,
    r_highlights: f32,
    g_shadows: f32,
    g_midtones: f32,
    g_highlights: f32,
    b_shadows: f32,
    b_midtones: f32,
    b_highlights: f32,
    // Global adjustments
    contrast: f32,
    saturation: f32,
    black_point: f32,  // Lift blacks (matte look)
    white_point: f32,  // Reduce whites
    gamma: f32,
}

fn build_channel_lut(
    shadows: f32,
    midtones: f32,
    highlights: f32,
    contrast: f32,
    black_point: f32,
    white_point: f32,
    gamma: f32,
) -> [u8; 256] {
    let mut lut = [0u8; 256];
    
    for i in 0..256 {
        let mut v = i as f32 / 255.0;
        
        // Apply gamma curve
        v = v.powf(gamma);
        
        // Apply tonal adjustments based on luminance zones
        let shadow_weight = (1.0 - v * 2.0).max(0.0);
        let highlight_weight = ((v - 0.5) * 2.0).max(0.0);
        let midtone_weight = 1.0 - (shadow_weight + highlight_weight).min(1.0);
        
        v += shadows * shadow_weight * 0.15;
        v += midtones * midtone_weight * 0.1;
        v += highlights * highlight_weight * 0.15;
        
        // Apply contrast
        v = (v - 0.5) * (1.0 + contrast * 0.01) + 0.5;
        
        // Apply black/white point (for matte/faded looks)
        v = black_point + v * (white_point - black_point);
        
        lut[i] = (v.clamp(0.0, 1.0) * 255.0) as u8;
    }
    
    lut
}

fn apply_film_profile(data: &[u8], profile: &FilmProfile) -> Vec<u8> {
    // Build per-channel LUTs
    let r_lut = build_channel_lut(
        profile.r_shadows, profile.r_midtones, profile.r_highlights,
        profile.contrast, profile.black_point, profile.white_point, profile.gamma
    );
    let g_lut = build_channel_lut(
        profile.g_shadows, profile.g_midtones, profile.g_highlights,
        profile.contrast, profile.black_point, profile.white_point, profile.gamma
    );
    let b_lut = build_channel_lut(
        profile.b_shadows, profile.b_midtones, profile.b_highlights,
        profile.contrast, profile.black_point, profile.white_point, profile.gamma
    );
    
    let mut result = data.to_vec();
    let saturation = profile.saturation;
    
    for chunk in result.chunks_exact_mut(4) {
        // Apply channel LUTs
        let r = r_lut[chunk[0] as usize] as f32;
        let g = g_lut[chunk[1] as usize] as f32;
        let b = b_lut[chunk[2] as usize] as f32;
        
        // Apply saturation
        let lum = r * 0.299 + g * 0.587 + b * 0.114;
        let r = lum + (r - lum) * saturation;
        let g = lum + (g - lum) * saturation;
        let b = lum + (b - lum) * saturation;
        
        chunk[0] = r.clamp(0.0, 255.0) as u8;
        chunk[1] = g.clamp(0.0, 255.0) as u8;
        chunk[2] = b.clamp(0.0, 255.0) as u8;
    }
    
    result
}

// ============ DETAIL (SHARPENING & NOISE REDUCTION) ============

/// Apply unsharp mask sharpening
/// amount: 0-150 (strength of sharpening)
/// radius: 0.5-3.0 (blur radius for the mask)
/// detail: 0-100 (how much to protect fine detail)
/// masking: 0-100 (edge masking - higher values sharpen only edges)
#[wasm_bindgen]
pub fn apply_sharpening(
    data: &[u8],
    width: u32,
    height: u32,
    amount: f32,
    radius: f32,
    detail: f32,
    masking: f32,
) -> Vec<u8> {
    if amount <= 0.0 {
        return data.to_vec();
    }
    
    let w = width as usize;
    let h = height as usize;
    let mut result = data.to_vec();
    
    // Create luminance buffer
    let mut lum: Vec<f32> = Vec::with_capacity(w * h);
    for chunk in data.chunks_exact(4) {
        let l = chunk[0] as f32 * 0.299 + chunk[1] as f32 * 0.587 + chunk[2] as f32 * 0.114;
        lum.push(l);
    }
    
    // Simple box blur approximation (faster than true Gaussian)
    let blur_radius = (radius * 2.0).max(1.0) as usize;
    let mut blurred = lum.clone();
    
    // Horizontal pass
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut count = 0;
            for dx in 0..=blur_radius * 2 {
                let nx = (x as i32 + dx as i32 - blur_radius as i32).clamp(0, w as i32 - 1) as usize;
                sum += blurred[y * w + nx];
                count += 1;
            }
            temp[y * w + x] = sum / count as f32;
        }
    }
    
    // Vertical pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut count = 0;
            for dy in 0..=blur_radius * 2 {
                let ny = (y as i32 + dy as i32 - blur_radius as i32).clamp(0, h as i32 - 1) as usize;
                sum += temp[ny * w + x];
                count += 1;
            }
            blurred[y * w + x] = sum / count as f32;
        }
    }
    
    // Calculate edge mask if masking > 0
    let mask_threshold = masking / 100.0;
    let detail_factor = 1.0 + detail / 50.0;
    let strength = amount / 100.0;
    
    // Apply unsharp mask
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pixel_idx = idx * 4;
            
            let orig_lum = lum[idx];
            let blur_lum = blurred[idx];
            let diff = orig_lum - blur_lum;
            
            // Calculate edge strength for masking
            let edge_strength = diff.abs() / 128.0;
            let mask = if mask_threshold > 0.0 {
                (edge_strength / mask_threshold).min(1.0)
            } else {
                1.0
            };
            
            // Apply detail protection (reduces sharpening in low-contrast areas)
            let detail_mask = (edge_strength * detail_factor).min(1.0);
            
            let final_strength = strength * mask * detail_mask;
            let sharpen_amount = diff * final_strength;
            
            // Apply to each channel proportionally
            for c in 0..3 {
                let orig = result[pixel_idx + c] as f32;
                let ratio = if orig_lum > 0.0 { orig / orig_lum } else { 1.0 };
                let new_val = orig + sharpen_amount * ratio;
                result[pixel_idx + c] = new_val.clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    result
}

/// Apply luminance noise reduction
/// luminance: 0-100 (strength of luminance noise reduction)
/// detail: 0-100 (preserve detail, higher = more detail preserved)
/// contrast: 0-100 (preserve contrast, higher = more contrast preserved)
#[wasm_bindgen]
pub fn apply_noise_reduction(
    data: &[u8],
    width: u32,
    height: u32,
    luminance: f32,
    lum_detail: f32,
    lum_contrast: f32,
    color: f32,
    color_detail: f32,
    smoothness: f32,
) -> Vec<u8> {
    if luminance <= 0.0 && color <= 0.0 {
        return data.to_vec();
    }
    
    let w = width as usize;
    let h = height as usize;
    let mut result = data.to_vec();
    
    let lum_strength = luminance / 100.0;
    let color_strength = color / 100.0;
    let detail_preserve = lum_detail / 100.0;
    let contrast_preserve = lum_contrast / 100.0;
    let smooth_factor = 1.0 + smoothness / 50.0;
    let color_detail_preserve = color_detail / 100.0;
    
    // Bilateral filter approximation for noise reduction
    let radius = ((lum_strength * 3.0).max(color_strength * 2.0) * smooth_factor) as usize;
    let radius = radius.clamp(1, 5);
    
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let center_r = data[idx] as f32;
            let center_g = data[idx + 1] as f32;
            let center_b = data[idx + 2] as f32;
            let center_lum = center_r * 0.299 + center_g * 0.587 + center_b * 0.114;
            
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            let mut weight_sum = 0.0;
            
            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let nidx = (ny * w + nx) * 4;
                    
                    let nr = data[nidx] as f32;
                    let ng = data[nidx + 1] as f32;
                    let nb = data[nidx + 2] as f32;
                    let nlum = nr * 0.299 + ng * 0.587 + nb * 0.114;
                    
                    // Spatial weight
                    let spatial_dist = ((dx * dx + dy * dy) as f32).sqrt();
                    let spatial_weight = (-spatial_dist / (radius as f32 + 1.0)).exp();
                    
                    // Luminance similarity weight (for detail preservation)
                    let lum_diff = (center_lum - nlum).abs();
                    let lum_weight = (-(lum_diff * lum_diff) / (50.0 * (1.0 - detail_preserve + 0.1))).exp();
                    
                    // Color similarity weight
                    let color_diff = ((center_r - nr).powi(2) + (center_g - ng).powi(2) + (center_b - nb).powi(2)).sqrt();
                    let color_weight = (-(color_diff * color_diff) / (100.0 * (1.0 - color_detail_preserve + 0.1))).exp();
                    
                    let weight = spatial_weight * lum_weight.powf(contrast_preserve) * color_weight;
                    
                    sum_r += nr * weight;
                    sum_g += ng * weight;
                    sum_b += nb * weight;
                    weight_sum += weight;
                }
            }
            
            if weight_sum > 0.0 {
                let new_r = sum_r / weight_sum;
                let new_g = sum_g / weight_sum;
                let new_b = sum_b / weight_sum;
                
                // Blend based on strength
                result[idx] = (center_r + (new_r - center_r) * lum_strength).clamp(0.0, 255.0) as u8;
                result[idx + 1] = (center_g + (new_g - center_g) * lum_strength.max(color_strength)).clamp(0.0, 255.0) as u8;
                result[idx + 2] = (center_b + (new_b - center_b) * lum_strength.max(color_strength)).clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    result
}

// ============ LENS CORRECTIONS ============

/// Apply lens distortion correction
/// distortion: -100 to 100 (negative = barrel, positive = pincushion)
#[wasm_bindgen]
pub fn apply_distortion(
    data: &[u8],
    width: u32,
    height: u32,
    distortion: f32,
) -> Vec<u8> {
    if distortion.abs() < 0.1 {
        return data.to_vec();
    }
    
    let w = width as usize;
    let h = height as usize;
    let mut result = vec![0u8; data.len()];
    
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let max_radius = (cx * cx + cy * cy).sqrt();
    let k = distortion / 1000.0; // Scale down for reasonable effect
    
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let r = (dx * dx + dy * dy).sqrt();
            let r_norm = r / max_radius;
            
            // Radial distortion formula
            let distort_factor = 1.0 + k * r_norm * r_norm;
            
            let src_x = cx + dx * distort_factor;
            let src_y = cy + dy * distort_factor;
            
            // Bilinear interpolation
            let src_x_int = src_x.floor() as i32;
            let src_y_int = src_y.floor() as i32;
            let fx = src_x - src_x_int as f32;
            let fy = src_y - src_y_int as f32;
            
            let dst_idx = (y * w + x) * 4;
            
            if src_x_int >= 0 && src_x_int < w as i32 - 1 && src_y_int >= 0 && src_y_int < h as i32 - 1 {
                for c in 0..4 {
                    let i00 = data[(src_y_int as usize * w + src_x_int as usize) * 4 + c] as f32;
                    let i10 = data[(src_y_int as usize * w + src_x_int as usize + 1) * 4 + c] as f32;
                    let i01 = data[((src_y_int + 1) as usize * w + src_x_int as usize) * 4 + c] as f32;
                    let i11 = data[((src_y_int + 1) as usize * w + src_x_int as usize + 1) * 4 + c] as f32;
                    
                    let val = i00 * (1.0 - fx) * (1.0 - fy) 
                            + i10 * fx * (1.0 - fy) 
                            + i01 * (1.0 - fx) * fy 
                            + i11 * fx * fy;
                    result[dst_idx + c] = val.clamp(0.0, 255.0) as u8;
                }
            } else {
                // Fill with edge color or transparent
                result[dst_idx + 3] = 255; // Keep alpha
            }
        }
    }
    
    result
}

/// Apply vignette effect
/// amount: -100 to 100 (negative = lighten edges, positive = darken edges)
/// midpoint: 0-100 (where vignette starts, 0 = center, 100 = edges)
/// roundness: -100 to 100 (negative = more rectangular, positive = more circular)
/// feather: 0-100 (how soft the vignette edge is)
#[wasm_bindgen]
pub fn apply_vignette(
    data: &[u8],
    width: u32,
    height: u32,
    amount: f32,
    midpoint: f32,
    roundness: f32,
    feather: f32,
) -> Vec<u8> {
    if amount.abs() < 0.1 {
        return data.to_vec();
    }
    
    let w = width as usize;
    let h = height as usize;
    let mut result = data.to_vec();
    
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let aspect = w as f32 / h as f32;
    
    let strength = -amount / 100.0; // Negative because positive amount should darken
    let mid = midpoint / 100.0;
    let round = 1.0 + roundness / 100.0;
    let soft = (feather / 100.0).max(0.01);
    
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f32 - cx) / cx;
            let dy = (y as f32 - cy) / cy;
            
            // Adjust for aspect ratio and roundness
            let dx_adj = dx * aspect.powf(round - 1.0);
            let dy_adj = dy;
            
            let dist = (dx_adj * dx_adj + dy_adj * dy_adj).sqrt();
            
            // Calculate vignette falloff
            let vignette = if dist < mid {
                1.0
            } else {
                let t = ((dist - mid) / soft).min(1.0);
                1.0 + strength * t * t
            };
            
            let idx = (y * w + x) * 4;
            for c in 0..3 {
                let val = result[idx + c] as f32 * vignette;
                result[idx + c] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    result
}

/// Apply defringe (chromatic aberration removal)
/// purple_amount: 0-20 (amount of purple fringe removal)
/// purple_hue: -100 to 100 (shift purple hue detection range)
/// green_amount: 0-20 (amount of green fringe removal)
/// green_hue: -100 to 100 (shift green hue detection range)
#[wasm_bindgen]
pub fn apply_defringe(
    data: &[u8],
    width: u32,
    height: u32,
    purple_amount: f32,
    purple_hue: f32,
    green_amount: f32,
    green_hue: f32,
) -> Vec<u8> {
    if purple_amount < 0.1 && green_amount < 0.1 {
        return data.to_vec();
    }
    
    let w = width as usize;
    let h = height as usize;
    let mut result = data.to_vec();
    
    let purple_str = purple_amount / 20.0;
    let green_str = green_amount / 20.0;
    
    // Purple hue center: ~280° (0.78 in 0-1)
    // Green hue center: ~120° (0.33 in 0-1)
    let purple_center = 0.78 + purple_hue / 360.0;
    let green_center = 0.33 + green_hue / 360.0;
    
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let r = result[idx] as f32 / 255.0;
            let g = result[idx + 1] as f32 / 255.0;
            let b = result[idx + 2] as f32 / 255.0;
            
            let (h, s, l) = rgb_to_hsl(r, g, b);
            
            // Check for purple fringe
            let purple_dist = ((h - purple_center).abs()).min((h - purple_center + 1.0).abs()).min((h - purple_center - 1.0).abs());
            if purple_dist < 0.15 && s > 0.2 && purple_str > 0.0 {
                // Desaturate purple fringes
                let factor = (1.0 - purple_dist / 0.15) * purple_str * s;
                let (new_r, new_g, new_b) = hsl_to_rgb(h, s * (1.0 - factor), l);
                result[idx] = (new_r * 255.0).clamp(0.0, 255.0) as u8;
                result[idx + 1] = (new_g * 255.0).clamp(0.0, 255.0) as u8;
                result[idx + 2] = (new_b * 255.0).clamp(0.0, 255.0) as u8;
            }
            
            // Check for green fringe
            let green_dist = ((h - green_center).abs()).min((h - green_center + 1.0).abs()).min((h - green_center - 1.0).abs());
            if green_dist < 0.15 && s > 0.2 && green_str > 0.0 {
                let factor = (1.0 - green_dist / 0.15) * green_str * s;
                let curr_r = result[idx] as f32 / 255.0;
                let curr_g = result[idx + 1] as f32 / 255.0;
                let curr_b = result[idx + 2] as f32 / 255.0;
                let (ch, cs, cl) = rgb_to_hsl(curr_r, curr_g, curr_b);
                let (new_r, new_g, new_b) = hsl_to_rgb(ch, cs * (1.0 - factor), cl);
                result[idx] = (new_r * 255.0).clamp(0.0, 255.0) as u8;
                result[idx + 1] = (new_g * 255.0).clamp(0.0, 255.0) as u8;
                result[idx + 2] = (new_b * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    
    result
}

#[wasm_bindgen]
pub fn apply_preset(data: &[u8], preset_id: u8) -> Vec<u8> {
    let profile = match preset_id {
        // Kodak Portra 400 - Warm, soft, lifted shadows, slightly desaturated
        0 => FilmProfile {
            r_shadows: 0.3, r_midtones: 0.1, r_highlights: 0.05,
            g_shadows: 0.1, g_midtones: 0.0, g_highlights: -0.05,
            b_shadows: -0.1, b_midtones: -0.1, b_highlights: -0.1,
            contrast: -5.0, saturation: 0.9, black_point: 0.02, white_point: 0.98, gamma: 1.05,
        },
        // Fuji Velvia 50 - Punchy, saturated, high contrast
        1 => FilmProfile {
            r_shadows: 0.0, r_midtones: 0.15, r_highlights: 0.1,
            g_shadows: 0.05, g_midtones: 0.1, g_highlights: 0.0,
            b_shadows: 0.1, b_midtones: 0.05, b_highlights: 0.15,
            contrast: 20.0, saturation: 1.4, black_point: 0.0, white_point: 1.0, gamma: 0.95,
        },
        // Kodak Gold 200 - Warm, golden tones, nostalgic
        2 => FilmProfile {
            r_shadows: 0.2, r_midtones: 0.15, r_highlights: 0.1,
            g_shadows: 0.1, g_midtones: 0.05, g_highlights: 0.0,
            b_shadows: -0.15, b_midtones: -0.1, b_highlights: -0.05,
            contrast: 5.0, saturation: 1.1, black_point: 0.01, white_point: 0.99, gamma: 1.0,
        },
        // Cinestill 800T - Tungsten, cyan shadows, halation look
        3 => FilmProfile {
            r_shadows: -0.1, r_midtones: 0.0, r_highlights: 0.15,
            g_shadows: 0.0, g_midtones: 0.0, g_highlights: 0.0,
            b_shadows: 0.2, b_midtones: 0.1, b_highlights: 0.0,
            contrast: 10.0, saturation: 1.05, black_point: 0.02, white_point: 0.97, gamma: 1.0,
        },
        // Kodak Ektar 100 - Vibrant, saturated, punchy
        4 => FilmProfile {
            r_shadows: 0.1, r_midtones: 0.1, r_highlights: 0.05,
            g_shadows: 0.0, g_midtones: 0.05, g_highlights: 0.0,
            b_shadows: 0.0, b_midtones: 0.0, b_highlights: 0.1,
            contrast: 15.0, saturation: 1.3, black_point: 0.0, white_point: 1.0, gamma: 0.98,
        },
        // Fuji Pro 400H - Soft, pastel, lifted shadows
        5 => FilmProfile {
            r_shadows: 0.1, r_midtones: 0.0, r_highlights: -0.05,
            g_shadows: 0.15, g_midtones: 0.05, g_highlights: 0.0,
            b_shadows: 0.1, b_midtones: 0.05, b_highlights: 0.05,
            contrast: -10.0, saturation: 0.85, black_point: 0.04, white_point: 0.98, gamma: 1.08,
        },
        // Ilford HP5 - Classic B&W, punchy
        6 => FilmProfile {
            r_shadows: 0.0, r_midtones: 0.0, r_highlights: 0.0,
            g_shadows: 0.0, g_midtones: 0.0, g_highlights: 0.0,
            b_shadows: 0.0, b_midtones: 0.0, b_highlights: 0.0,
            contrast: 25.0, saturation: 0.0, black_point: 0.02, white_point: 0.98, gamma: 1.0,
        },
        // Kodak T-Max 400 - Smooth B&W, wide tonal range
        7 => FilmProfile {
            r_shadows: 0.0, r_midtones: 0.0, r_highlights: 0.0,
            g_shadows: 0.0, g_midtones: 0.0, g_highlights: 0.0,
            b_shadows: 0.0, b_midtones: 0.0, b_highlights: 0.0,
            contrast: 15.0, saturation: 0.0, black_point: 0.03, white_point: 0.97, gamma: 1.05,
        },
        // Matte Film - Faded blacks, low contrast
        8 => FilmProfile {
            r_shadows: 0.1, r_midtones: 0.0, r_highlights: -0.05,
            g_shadows: 0.1, g_midtones: 0.0, g_highlights: -0.05,
            b_shadows: 0.15, b_midtones: 0.0, b_highlights: -0.05,
            contrast: -15.0, saturation: 0.9, black_point: 0.08, white_point: 0.95, gamma: 1.1,
        },
        // Faded Vintage - Washed out, warm
        9 => FilmProfile {
            r_shadows: 0.2, r_midtones: 0.1, r_highlights: 0.0,
            g_shadows: 0.1, g_midtones: 0.05, g_highlights: -0.05,
            b_shadows: 0.0, b_midtones: -0.05, b_highlights: -0.1,
            contrast: -20.0, saturation: 0.75, black_point: 0.1, white_point: 0.92, gamma: 1.15,
        },
        // High Contrast - Punchy, dramatic
        10 => FilmProfile {
            r_shadows: 0.0, r_midtones: 0.0, r_highlights: 0.0,
            g_shadows: 0.0, g_midtones: 0.0, g_highlights: 0.0,
            b_shadows: 0.0, b_midtones: 0.0, b_highlights: 0.0,
            contrast: 40.0, saturation: 1.15, black_point: 0.0, white_point: 1.0, gamma: 0.95,
        },
        // Soft & Dreamy - Bright, airy, low contrast
        11 => FilmProfile {
            r_shadows: 0.15, r_midtones: 0.05, r_highlights: 0.0,
            g_shadows: 0.1, g_midtones: 0.05, g_highlights: 0.0,
            b_shadows: 0.1, b_midtones: 0.0, b_highlights: 0.05,
            contrast: -25.0, saturation: 0.85, black_point: 0.06, white_point: 1.0, gamma: 1.2,
        },
        // Default - no change
        _ => FilmProfile {
            r_shadows: 0.0, r_midtones: 0.0, r_highlights: 0.0,
            g_shadows: 0.0, g_midtones: 0.0, g_highlights: 0.0,
            b_shadows: 0.0, b_midtones: 0.0, b_highlights: 0.0,
            contrast: 0.0, saturation: 1.0, black_point: 0.0, white_point: 1.0, gamma: 1.0,
        },
    };
    
    apply_film_profile(data, &profile)
}
