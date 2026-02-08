# OpenRoom

A Lightroom-inspired photo editor that runs entirely in your browser. Built with Rust compiled to WebAssembly for fast image processing and React for the user interface.

## Overview

OpenRoom brings professional photo editing capabilities to the browser, offering a Lightroom-like experience without requiring any software installation. It processes images using Rust code compiled to WebAssembly, delivering near-native performance for computationally intensive operations like noise reduction, sharpening, and tone curve adjustments.

## Features

### Basic Adjustments
- Exposure control (-5 to +5 stops)
- Contrast adjustment
- Highlights and shadows recovery
- Whites and blacks level adjustment

### HSL Color Grading
Fine-tune hue, saturation, and luminance for eight color channels:
- Red, Orange, Yellow, Green
- Aqua, Blue, Purple, Magenta

### Detail Enhancements
- **Sharpening**: Amount, radius, detail, and masking controls
- **Noise Reduction**: Luminance and color noise reduction with detail preservation
- **Film Grain**: Adjustable amount, size, and roughness for analog texture

### Lens Corrections
- Barrel and pincushion distortion correction
- Vignette removal or addition with midpoint, roundness, and feather controls
- Chromatic aberration correction (defringe) for purple and green fringes

### Tone Curves
- Full RGB curve control
- Individual channel curves (Red, Green, Blue)
- Interactive curve editor with smooth spline interpolation
- Multiple control points for precise tonal adjustments

### Film Presets
One-click film emulation presets including:
- Kodak Portra 400
- Fuji Velvia 50
- Kodak Gold 200
- Cinestill 800T
- Kodak Ektar 100
- Fuji Pro 400H
- Ilford HP5
- Kodak T-Max 400
- Plus creative presets (Matte Film, Faded Vintage, High Contrast, Soft & Dreamy)

## Architecture

The application uses a hybrid architecture:

- **Frontend**: React 19 with TypeScript, Tailwind CSS for styling, Vite for build tooling
- **Image Processing**: Rust compiled to WebAssembly (WASM) for performance-critical operations
- **Communication**: React components call WASM functions directly for image transformations

This architecture delivers professional-grade image processing speeds in the browser without sacrificing the responsive user experience of a modern web application.

## Project Structure

```
openroom/
├── frontend/           # React application
│   ├── src/
│   │   ├── App.tsx    # Main application component with all editing controls
│   │   ├── wasm/      # WASM bindings and generated files
│   │   └── main.tsx   # Application entry point
│   └── package.json
├── backend/           # Rust image processing library
│   └── src/lib.rs     # Core image processing functions
└── README.md
```

## Development

### Prerequisites
- Node.js and pnpm (for frontend)
- Rust toolchain (for backend)

### Frontend Setup

```bash
cd frontend
pnpm install
pnpm dev
```

The development server will start on `http://localhost:5173`.

### Backend Setup

```bash
cd backend
cargo build
wasm-pack build --target web --out-dir ../frontend/src/wasm
```

After building the WASM module, the frontend can import and use the Rust functions.

### Build for Production

Frontend:
```bash
cd frontend
pnpm build
```

Backend (rebuild WASM when Rust code changes):
```bash
cd backend
wasm-pack build --target web --out-dir ../frontend/src/wasm
```

## Available Scripts

**Frontend:**
- `pnpm dev` - Start development server
- `pnpm build` - Build for production
- `pnpm lint` - Run ESLint
- `pnpm preview` - Preview production build

**Backend:**
- `cargo build` - Build Rust library
- `wasm-pack build --target web --out-dir ../frontend/src/wasm` - Build WASM module

## Technical Details

The Rust backend implements several optimized image processing algorithms:

- **Tone Curves**: Uses cubic spline interpolation with pre-computed lookup tables (LUTs) for maximum speed
- **HSL Adjustments**: Converts RGB to HSL color space, applies adjustments, and converts back with smooth color transitions
- **Sharpening**: Implements unsharp mask algorithm with radius, detail, and masking controls
- **Noise Reduction**: Edge-preserving smoothing algorithms for luminance and chroma noise
- **Grain**: Procedural film grain generation using pseudo-random number generators
- **Distortion**: Geometric transformation with bilinear interpolation
- **Vignette**: Radial gradient application based on distance from image center
- **Defringe**: Selective color replacement to remove chromatic aberrations

All processing is done on raw image data (RGBA bytes) passed between JavaScript and WebAssembly for minimal overhead.
