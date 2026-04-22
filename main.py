import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
from flask import Flask, request, render_template_string, send_from_directory, url_for,jsonify
from PIL import Image
from dotenv import load_dotenv
from app.model_service import predict_ensemble
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ----------------------------
# HTML Templates
# ----------------------------
UPLOAD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JAAM AI - MRI Analysis Upload</title>
  <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>

  <!-- ✅ Inline CSS -->
  <style>
   :root {
  /* Primitive Color Tokens */
  --color-white: rgba(255, 255, 255, 1);
  --color-black: rgba(0, 0, 0, 1);
  --color-cream-50: rgba(252, 252, 249, 1);
  --color-cream-100: rgba(255, 255, 253, 1);
  --color-gray-200: rgba(245, 245, 245, 1);
  --color-gray-300: rgba(167, 169, 169, 1);
  --color-gray-400: rgba(119, 124, 124, 1);
  --color-slate-500: rgba(98, 108, 113, 1);
  --color-brown-600: rgba(94, 82, 64, 1);
  --color-charcoal-700: rgba(31, 33, 33, 1);
  --color-charcoal-800: rgba(38, 40, 40, 1);
  --color-slate-900: rgba(19, 52, 59, 1);
  --color-teal-300: rgba(50, 184, 198, 1);
  --color-teal-400: rgba(45, 166, 178, 1);
  --color-teal-500: rgba(33, 128, 141, 1);
  --color-teal-600: rgba(29, 116, 128, 1);
  --color-teal-700: rgba(26, 104, 115, 1);
  --color-teal-800: rgba(41, 150, 161, 1);
  --color-red-400: rgba(255, 84, 89, 1);
  --color-red-500: rgba(192, 21, 47, 1);
  --color-orange-400: rgba(230, 129, 97, 1);
  --color-orange-500: rgba(168, 75, 47, 1);

  /* RGB versions for opacity control */
  --color-brown-600-rgb: 94, 82, 64;
  --color-teal-500-rgb: 33, 128, 141;
  --color-slate-900-rgb: 19, 52, 59;
  --color-slate-500-rgb: 98, 108, 113;
  --color-red-500-rgb: 192, 21, 47;
  --color-red-400-rgb: 255, 84, 89;
  --color-orange-500-rgb: 168, 75, 47;
  --color-orange-400-rgb: 230, 129, 97;

  /* Background color tokens (Light Mode) */
  --color-bg-1: rgba(59, 130, 246, 0.08); /* Light blue */
  --color-bg-2: rgba(245, 158, 11, 0.08); /* Light yellow */
  --color-bg-3: rgba(34, 197, 94, 0.08); /* Light green */
  --color-bg-4: rgba(239, 68, 68, 0.08); /* Light red */
  --color-bg-5: rgba(147, 51, 234, 0.08); /* Light purple */
  --color-bg-6: rgba(249, 115, 22, 0.08); /* Light orange */
  --color-bg-7: rgba(236, 72, 153, 0.08); /* Light pink */
  --color-bg-8: rgba(6, 182, 212, 0.08); /* Light cyan */

  /* Semantic Color Tokens (Light Mode) */
  --color-background: var(--color-cream-50);
  --color-surface: var(--color-cream-100);
  --color-text: var(--color-slate-900);
  --color-text-secondary: var(--color-slate-500);
  --color-primary: var(--color-teal-500);
  --color-primary-hover: var(--color-teal-600);
  --color-primary-active: var(--color-teal-700);
  --color-secondary: rgba(var(--color-brown-600-rgb), 0.12);
  --color-secondary-hover: rgba(var(--color-brown-600-rgb), 0.2);
  --color-secondary-active: rgba(var(--color-brown-600-rgb), 0.25);
  --color-border: rgba(var(--color-brown-600-rgb), 0.2);
  --color-btn-primary-text: var(--color-cream-50);
  --color-card-border: rgba(var(--color-brown-600-rgb), 0.12);
  --color-card-border-inner: rgba(var(--color-brown-600-rgb), 0.12);
  --color-error: var(--color-red-500);
  --color-success: var(--color-teal-500);
  --color-warning: var(--color-orange-500);
  --color-info: var(--color-slate-500);
  --color-focus-ring: rgba(var(--color-teal-500-rgb), 0.4);
  --color-select-caret: rgba(var(--color-slate-900-rgb), 0.8);

  /* Common style patterns */
  --focus-ring: 0 0 0 3px var(--color-focus-ring);
  --focus-outline: 2px solid var(--color-primary);
  --status-bg-opacity: 0.15;
  --status-border-opacity: 0.25;
  --select-caret-light: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23134252' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  --select-caret-dark: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f5f5f5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");

  /* RGB versions for opacity control */
  --color-success-rgb: 33, 128, 141;
  --color-error-rgb: 192, 21, 47;
  --color-warning-rgb: 168, 75, 47;
  --color-info-rgb: 98, 108, 113;

  /* Typography */
  --font-family-base: "FKGroteskNeue", "Geist", "Inter", -apple-system,
    BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-family-mono: "Berkeley Mono", ui-monospace, SFMono-Regular, Menlo,
    Monaco, Consolas, monospace;
  --font-size-xs: 11px;
  --font-size-sm: 12px;
  --font-size-base: 14px;
  --font-size-md: 14px;
  --font-size-lg: 16px;
  --font-size-xl: 18px;
  --font-size-2xl: 20px;
  --font-size-3xl: 24px;
  --font-size-4xl: 30px;
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 550;
  --font-weight-bold: 600;
  --line-height-tight: 1.2;
  --line-height-normal: 1.5;
  --letter-spacing-tight: -0.01em;

  /* Spacing */
  --space-0: 0;
  --space-1: 1px;
  --space-2: 2px;
  --space-4: 4px;
  --space-6: 6px;
  --space-8: 8px;
  --space-10: 10px;
  --space-12: 12px;
  --space-16: 16px;
  --space-20: 20px;
  --space-24: 24px;
  --space-32: 32px;

  /* Border Radius */
  --radius-sm: 6px;
  --radius-base: 8px;
  --radius-md: 10px;
  --radius-lg: 12px;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.02);
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.04),
    0 2px 4px -1px rgba(0, 0, 0, 0.02);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.04),
    0 4px 6px -2px rgba(0, 0, 0, 0.02);
  --shadow-inset-sm: inset 0 1px 0 rgba(255, 255, 255, 0.15),
    inset 0 -1px 0 rgba(0, 0, 0, 0.03);

  /* Animation */
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --ease-standard: cubic-bezier(0.16, 1, 0.3, 1);

  /* Layout */
  --container-sm: 640px;
  --container-md: 768px;
  --container-lg: 1024px;
  --container-xl: 1280px;
}

/* Dark mode colors */
@media (prefers-color-scheme: dark) {
  :root {
    /* RGB versions for opacity control (Dark Mode) */
    --color-gray-400-rgb: 119, 124, 124;
    --color-teal-300-rgb: 50, 184, 198;
    --color-gray-300-rgb: 167, 169, 169;
    --color-gray-200-rgb: 245, 245, 245;

    /* Background color tokens (Dark Mode) */
    --color-bg-1: rgba(29, 78, 216, 0.15); /* Dark blue */
    --color-bg-2: rgba(180, 83, 9, 0.15); /* Dark yellow */
    --color-bg-3: rgba(21, 128, 61, 0.15); /* Dark green */
    --color-bg-4: rgba(185, 28, 28, 0.15); /* Dark red */
    --color-bg-5: rgba(107, 33, 168, 0.15); /* Dark purple */
    --color-bg-6: rgba(194, 65, 12, 0.15); /* Dark orange */
    --color-bg-7: rgba(190, 24, 93, 0.15); /* Dark pink */
    --color-bg-8: rgba(8, 145, 178, 0.15); /* Dark cyan */

    /* Semantic Color Tokens (Dark Mode) */
    --color-background: var(--color-charcoal-700);
    --color-surface: var(--color-charcoal-800);
    --color-text: var(--color-gray-200);
    --color-text-secondary: rgba(var(--color-gray-300-rgb), 0.7);
    --color-primary: var(--color-teal-300);
    --color-primary-hover: var(--color-teal-400);
    --color-primary-active: var(--color-teal-800);
    --color-secondary: rgba(var(--color-gray-400-rgb), 0.15);
    --color-secondary-hover: rgba(var(--color-gray-400-rgb), 0.25);
    --color-secondary-active: rgba(var(--color-gray-400-rgb), 0.3);
    --color-border: rgba(var(--color-gray-400-rgb), 0.3);
    --color-error: var(--color-red-400);
    --color-success: var(--color-teal-300);
    --color-warning: var(--color-orange-400);
    --color-info: var(--color-gray-300);
    --color-focus-ring: rgba(var(--color-teal-300-rgb), 0.4);
    --color-btn-primary-text: var(--color-slate-900);
    --color-card-border: rgba(var(--color-gray-400-rgb), 0.2);
    --color-card-border-inner: rgba(var(--color-gray-400-rgb), 0.15);
    --shadow-inset-sm: inset 0 1px 0 rgba(255, 255, 255, 0.1),
      inset 0 -1px 0 rgba(0, 0, 0, 0.15);
    --button-border-secondary: rgba(var(--color-gray-400-rgb), 0.2);
    --color-border-secondary: rgba(var(--color-gray-400-rgb), 0.2);
    --color-select-caret: rgba(var(--color-gray-200-rgb), 0.8);

    /* Common style patterns - updated for dark mode */
    --focus-ring: 0 0 0 3px var(--color-focus-ring);
    --focus-outline: 2px solid var(--color-primary);
    --status-bg-opacity: 0.15;
    --status-border-opacity: 0.25;
    --select-caret-light: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23134252' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    --select-caret-dark: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f5f5f5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");

    /* RGB versions for dark mode */
    --color-success-rgb: var(--color-teal-300-rgb);
    --color-error-rgb: var(--color-red-400-rgb);
    --color-warning-rgb: var(--color-orange-400-rgb);
    --color-info-rgb: var(--color-gray-300-rgb);
  }
}

/* Data attribute for manual theme switching */
[data-color-scheme="dark"] {
  /* RGB versions for opacity control (dark mode) */
  --color-gray-400-rgb: 119, 124, 124;
  --color-teal-300-rgb: 50, 184, 198;
  --color-gray-300-rgb: 167, 169, 169;
  --color-gray-200-rgb: 245, 245, 245;

  /* Colorful background palette - Dark Mode */
  --color-bg-1: rgba(29, 78, 216, 0.15); /* Dark blue */
  --color-bg-2: rgba(180, 83, 9, 0.15); /* Dark yellow */
  --color-bg-3: rgba(21, 128, 61, 0.15); /* Dark green */
  --color-bg-4: rgba(185, 28, 28, 0.15); /* Dark red */
  --color-bg-5: rgba(107, 33, 168, 0.15); /* Dark purple */
  --color-bg-6: rgba(194, 65, 12, 0.15); /* Dark orange */
  --color-bg-7: rgba(190, 24, 93, 0.15); /* Dark pink */
  --color-bg-8: rgba(8, 145, 178, 0.15); /* Dark cyan */

  /* Semantic Color Tokens (Dark Mode) */
  --color-background: var(--color-charcoal-700);
  --color-surface: var(--color-charcoal-800);
  --color-text: var(--color-gray-200);
  --color-text-secondary: rgba(var(--color-gray-300-rgb), 0.7);
  --color-primary: var(--color-teal-300);
  --color-primary-hover: var(--color-teal-400);
  --color-primary-active: var(--color-teal-800);
  --color-secondary: rgba(var(--color-gray-400-rgb), 0.15);
  --color-secondary-hover: rgba(var(--color-gray-400-rgb), 0.25);
  --color-secondary-active: rgba(var(--color-gray-400-rgb), 0.3);
  --color-border: rgba(var(--color-gray-400-rgb), 0.3);
  --color-error: var(--color-red-400);
  --color-success: var(--color-teal-300);
  --color-warning: var(--color-orange-400);
  --color-info: var(--color-gray-300);
  --color-focus-ring: rgba(var(--color-teal-300-rgb), 0.4);
  --color-btn-primary-text: var(--color-slate-900);
  --color-card-border: rgba(var(--color-gray-400-rgb), 0.15);
  --color-card-border-inner: rgba(var(--color-gray-400-rgb), 0.15);
  --shadow-inset-sm: inset 0 1px 0 rgba(255, 255, 255, 0.1),
    inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  --color-border-secondary: rgba(var(--color-gray-400-rgb), 0.2);
  --color-select-caret: rgba(var(--color-gray-200-rgb), 0.8);

  /* Common style patterns - updated for dark mode */
  --focus-ring: 0 0 0 3px var(--color-focus-ring);
  --focus-outline: 2px solid var(--color-primary);
  --status-bg-opacity: 0.15;
  --status-border-opacity: 0.25;
  --select-caret-light: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23134252' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  --select-caret-dark: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f5f5f5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");

  /* RGB versions for dark mode */
  --color-success-rgb: var(--color-teal-300-rgb);
  --color-error-rgb: var(--color-red-400-rgb);
  --color-warning-rgb: var(--color-orange-400-rgb);
  --color-info-rgb: var(--color-gray-300-rgb);
}

[data-color-scheme="light"] {
  /* RGB versions for opacity control (light mode) */
  --color-brown-600-rgb: 94, 82, 64;
  --color-teal-500-rgb: 33, 128, 141;
  --color-slate-900-rgb: 19, 52, 59;

  /* Semantic Color Tokens (Light Mode) */
  --color-background: var(--color-cream-50);
  --color-surface: var(--color-cream-100);
  --color-text: var(--color-slate-900);
  --color-text-secondary: var(--color-slate-500);
  --color-primary: var(--color-teal-500);
  --color-primary-hover: var(--color-teal-600);
  --color-primary-active: var(--color-teal-700);
  --color-secondary: rgba(var(--color-brown-600-rgb), 0.12);
  --color-secondary-hover: rgba(var(--color-brown-600-rgb), 0.2);
  --color-secondary-active: rgba(var(--color-brown-600-rgb), 0.25);
  --color-border: rgba(var(--color-brown-600-rgb), 0.2);
  --color-btn-primary-text: var(--color-cream-50);
  --color-card-border: rgba(var(--color-brown-600-rgb), 0.12);
  --color-card-border-inner: rgba(var(--color-brown-600-rgb), 0.12);
  --color-error: var(--color-red-500);
  --color-success: var(--color-teal-500);
  --color-warning: var(--color-orange-500);
  --color-info: var(--color-slate-500);
  --color-focus-ring: rgba(var(--color-teal-500-rgb), 0.4);

  /* RGB versions for light mode */
  --color-success-rgb: var(--color-teal-500-rgb);
  --color-error-rgb: var(--color-red-500-rgb);
  --color-warning-rgb: var(--color-orange-500-rgb);
  --color-info-rgb: var(--color-slate-500-rgb);
}

/* Base styles */
html {
  font-size: var(--font-size-base);
  font-family: var(--font-family-base);
  line-height: var(--line-height-normal);
  color: var(--color-text);
  background-color: var(--color-background);
  -webkit-font-smoothing: antialiased;
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
}

*,
*::before,
*::after {
  box-sizing: inherit;
}

/* Typography */
h1,
h2,
h3,
h4,
h5,
h6 {
  margin: 0;
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-tight);
  color: var(--color-text);
  letter-spacing: var(--letter-spacing-tight);
}

h1 {
  font-size: var(--font-size-4xl);
}
h2 {
  font-size: var(--font-size-3xl);
}
h3 {
  font-size: var(--font-size-2xl);
}
h4 {
  font-size: var(--font-size-xl);
}
h5 {
  font-size: var(--font-size-lg);
}
h6 {
  font-size: var(--font-size-md);
}

p {
  margin: 0 0 var(--space-16) 0;
}

a {
  color: var(--color-primary);
  text-decoration: none;
  transition: color var(--duration-fast) var(--ease-standard);
}

a:hover {
  color: var(--color-primary-hover);
}

code,
pre {
  font-family: var(--font-family-mono);
  font-size: calc(var(--font-size-base) * 0.95);
  background-color: var(--color-secondary);
  border-radius: var(--radius-sm);
}

code {
  padding: var(--space-1) var(--space-4);
}

pre {
  padding: var(--space-16);
  margin: var(--space-16) 0;
  overflow: auto;
  border: 1px solid var(--color-border);
}

pre code {
  background: none;
  padding: 0;
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-8) var(--space-16);
  border-radius: var(--radius-base);
  font-size: var(--font-size-base);
  font-weight: 500;
  line-height: 1.5;
  cursor: pointer;
  transition: all var(--duration-normal) var(--ease-standard);
  border: none;
  text-decoration: none;
  position: relative;
}

.btn:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}

.btn--primary {
  background: var(--color-primary);
  color: var(--color-btn-primary-text);
}

.btn--primary:hover {
  background: var(--color-primary-hover);
}

.btn--primary:active {
  background: var(--color-primary-active);
}

.btn--secondary {
  background: var(--color-secondary);
  color: var(--color-text);
}

.btn--secondary:hover {
  background: var(--color-secondary-hover);
}

.btn--secondary:active {
  background: var(--color-secondary-active);
}

.btn--outline {
  background: transparent;
  border: 1px solid var(--color-border);
  color: var(--color-text);
}

.btn--outline:hover {
  background: var(--color-secondary);
}

.btn--sm {
  padding: var(--space-4) var(--space-12);
  font-size: var(--font-size-sm);
  border-radius: var(--radius-sm);
}

.btn--lg {
  padding: var(--space-10) var(--space-20);
  font-size: var(--font-size-lg);
  border-radius: var(--radius-md);
}

.btn--full-width {
  width: 100%;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Form elements */
.form-control {
  display: block;
  width: 100%;
  padding: var(--space-8) var(--space-12);
  font-size: var(--font-size-md);
  line-height: 1.5;
  color: var(--color-text);
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-base);
  transition: border-color var(--duration-fast) var(--ease-standard),
    box-shadow var(--duration-fast) var(--ease-standard);
}

textarea.form-control {
  font-family: var(--font-family-base);
  font-size: var(--font-size-base);
}

select.form-control {
  padding: var(--space-8) var(--space-12);
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  background-image: var(--select-caret-light);
  background-repeat: no-repeat;
  background-position: right var(--space-12) center;
  background-size: 16px;
  padding-right: var(--space-32);
}

/* Add a dark mode specific caret */
@media (prefers-color-scheme: dark) {
  select.form-control {
    background-image: var(--select-caret-dark);
  }
}

/* Also handle data-color-scheme */
[data-color-scheme="dark"] select.form-control {
  background-image: var(--select-caret-dark);
}

[data-color-scheme="light"] select.form-control {
  background-image: var(--select-caret-light);
}

.form-control:focus {
  border-color: var(--color-primary);
  outline: var(--focus-outline);
}

.form-label {
  display: block;
  margin-bottom: var(--space-8);
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
}

.form-group {
  margin-bottom: var(--space-16);
}

/* Card component */
.card {
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-card-border);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  transition: box-shadow var(--duration-normal) var(--ease-standard);
}

.card:hover {
  box-shadow: var(--shadow-md);
}

.card__body {
  padding: var(--space-16);
}

.card__header,
.card__footer {
  padding: var(--space-16);
  border-bottom: 1px solid var(--color-card-border-inner);
}

/* Status indicators - simplified with CSS variables */
.status {
  display: inline-flex;
  align-items: center;
  padding: var(--space-6) var(--space-12);
  border-radius: var(--radius-full);
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
}

.status--success {
  background-color: rgba(
    var(--color-success-rgb, 33, 128, 141),
    var(--status-bg-opacity)
  );
  color: var(--color-success);
  border: 1px solid
    rgba(var(--color-success-rgb, 33, 128, 141), var(--status-border-opacity));
}

.status--error {
  background-color: rgba(
    var(--color-error-rgb, 192, 21, 47),
    var(--status-bg-opacity)
  );
  color: var(--color-error);
  border: 1px solid
    rgba(var(--color-error-rgb, 192, 21, 47), var(--status-border-opacity));
}

.status--warning {
  background-color: rgba(
    var(--color-warning-rgb, 168, 75, 47),
    var(--status-bg-opacity)
  );
  color: var(--color-warning);
  border: 1px solid
    rgba(var(--color-warning-rgb, 168, 75, 47), var(--status-border-opacity));
}

.status--info {
  background-color: rgba(
    var(--color-info-rgb, 98, 108, 113),
    var(--status-bg-opacity)
  );
  color: var(--color-info);
  border: 1px solid
    rgba(var(--color-info-rgb, 98, 108, 113), var(--status-border-opacity));
}

/* Container layout */
.container {
  width: 100%;
  margin-right: auto;
  margin-left: auto;
  padding-right: var(--space-16);
  padding-left: var(--space-16);
}

@media (min-width: 640px) {
  .container {
    max-width: var(--container-sm);
  }
}
@media (min-width: 768px) {
  .container {
    max-width: var(--container-md);
  }
}
@media (min-width: 1024px) {
  .container {
    max-width: var(--container-lg);
  }
}
@media (min-width: 1280px) {
  .container {
    max-width: var(--container-xl);
  }
}

/* Utility classes */
.flex {
  display: flex;
}
.flex-col {
  flex-direction: column;
}
.items-center {
  align-items: center;
}
.justify-center {
  justify-content: center;
}
.justify-between {
  justify-content: space-between;
}
.gap-4 {
  gap: var(--space-4);
}
.gap-8 {
  gap: var(--space-8);
}
.gap-16 {
  gap: var(--space-16);
}

.m-0 {
  margin: 0;
}
.mt-8 {
  margin-top: var(--space-8);
}
.mb-8 {
  margin-bottom: var(--space-8);
}
.mx-8 {
  margin-left: var(--space-8);
  margin-right: var(--space-8);
}
.my-8 {
  margin-top: var(--space-8);
  margin-bottom: var(--space-8);
}

.p-0 {
  padding: 0;
}
.py-8 {
  padding-top: var(--space-8);
  padding-bottom: var(--space-8);
}
.px-8 {
  padding-left: var(--space-8);
  padding-right: var(--space-8);
}
.py-16 {
  padding-top: var(--space-16);
  padding-bottom: var(--space-16);
}
.px-16 {
  padding-left: var(--space-16);
  padding-right: var(--space-16);
}

.block {
  display: block;
}
.hidden {
  display: none;
}

/* Accessibility */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

:focus-visible {
  outline: var(--focus-outline);
  outline-offset: 2px;
}

/* Dark mode specifics */
[data-color-scheme="dark"] .btn--outline {
  border: 1px solid var(--color-border-secondary);
}

@font-face {
  font-family: 'FKGroteskNeue';
  src: url('https://r2cdn.perplexity.ai/fonts/FKGroteskNeue.woff2')
    format('woff2');
}

/* END PERPLEXITY DESIGN SYSTEM */
/* Blue Dark Theme Variables */
:root {
  /* Primary blue theme colors */
  --bg-primary: #0f1629;
  --bg-secondary: #1e293b;
  --card-bg: #1e3a5f;
  --card-bg-hover: #2d4a6b;
  --border-color: #3b4d66;
  --border-active: #3b82f6;
  --text-primary: #e2e8f0;
  --text-secondary: #cbd5e1;
  --accent-blue: #3b82f6;
  --accent-light: #60a5fa;
  --success-color: #10b981;
  --warning-color: #f59e0b;
  --danger-color: #ef4444;

  /* Spacing and sizing */
  --space-4: 4px;
  --space-8: 8px;
  --space-12: 12px;
  --space-16: 16px;
  --space-20: 20px;
  --space-24: 24px;
  --space-32: 32px;
  --space-48: 48px;
  --space-64: 64px;

  /* Border radius */
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;

  /* Transitions */
  --transition: all 0.3s ease;
  --transition-fast: all 0.15s ease;
}

/* Base styles */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

*:focus {
  outline: none;
}

*:focus-visible {
  outline: 2px solid var(--accent-blue);
  outline-offset: 2px;
}

html {
  font-size: 16px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: radial-gradient(circle at 50% 30%, var(--card-bg) 0%, var(--bg-primary) 50%, #0a0f1a 100%);
  color: var(--text-primary);
  min-height: 100vh;
  line-height: 1.6;
  overflow-x: hidden;
}

.app-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--space-32) var(--space-16);
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
}

/* Logo Section */
.logo-section {
  text-align: center;
  margin-bottom: var(--space-64);
  animation: fadeInDown 0.8s ease-out;
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-16);
  margin-bottom: var(--space-12);
}

.logo-icon {
  width: 48px;
  height: 48px;
  color: var(--accent-blue);
  filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.4));
}

.logo-text {
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.025em;
}

.logo-subtitle {
  font-size: 1.1rem;
  color: var(--text-secondary);
  font-weight: 400;
}

/* Upload Section */
.upload-section {
  width: 100%;
  max-width: 600px;
  animation: fadeInUp 0.8s ease-out 0.2s both;
}

.upload-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-xl);
  padding: var(--space-48);
  box-shadow: 
    0 20px 25px -5px rgba(0, 0, 0, 0.3),
    0 10px 10px -5px rgba(0, 0, 0, 0.2);
  transition: var(--transition);
  position: relative;
  overflow: hidden;
}

.upload-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
  opacity: 0.6;
}

.upload-card:hover {
  background: var(--card-bg-hover);
  transform: translateY(-2px);
  box-shadow: 
    0 25px 50px -12px rgba(0, 0, 0, 0.4),
    0 0 30px rgba(59, 130, 246, 0.1);
}

/* Upload Area States */
.upload-area {
  position: relative;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-content {
  text-align: center;
  padding: var(--space-32);
  border: 2px dashed var(--border-color);
  border-radius: var(--radius-lg);
  transition: var(--transition);
  cursor: pointer;
  width: 100%;
  position: relative;
}

.upload-content:hover,
.upload-content.drag-over {
  border-color: var(--accent-blue);
  background: rgba(59, 130, 246, 0.05);
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.1);
}

.upload-content.drag-over {
  transform: scale(1.02);
  border-style: solid;
  background: rgba(59, 130, 246, 0.1);
}

.upload-icon {
  width: 64px;
  height: 64px;
  color: var(--accent-blue);
  margin-bottom: var(--space-24);
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}

.upload-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--space-12);
}

.upload-subtitle {
  font-size: 1rem;
  color: var(--text-secondary);
  margin-bottom: var(--space-24);
}

.upload-info {
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-top: var(--space-24);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-8);
  padding: var(--space-12) var(--space-24);
  border: none;
  border-radius: var(--radius-md);
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
  text-decoration: none;
  position: relative;
  overflow: hidden;
}

.btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  transition: left 0.5s;
  pointer-events: none;
}

.btn:hover::before {
  left: 100%;
}

.btn:focus-visible {
  outline: 2px solid var(--accent-blue);
  outline-offset: 2px;
}

.btn-primary {
  background: var(--accent-blue);
  color: white;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

.btn-primary:hover {
  background: var(--accent-light);
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
}

.btn-secondary {
  background: rgba(203, 213, 225, 0.1);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
}

.btn-secondary:hover {
  background: rgba(203, 213, 225, 0.2);
  border-color: var(--accent-blue);
}

.btn-small {
  padding: var(--space-8) var(--space-16);
  font-size: 0.875rem;
}

.select-file-btn {
  background: rgba(59, 130, 246, 0.1);
  color: var(--accent-blue);
  border: 1px solid var(--accent-blue);
  padding: var(--space-16) var(--space-32);
  border-radius: var(--radius-md);
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition);
  display: inline-flex;
  align-items: center;
  gap: var(--space-12);
  position: relative;
  overflow: hidden;
}

.select-file-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
  transition: left 0.5s;
  pointer-events: none;
}

.select-file-btn:hover::before {
  left: 100%;
}

.select-file-btn:hover {
  background: var(--accent-blue);
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
}

.select-file-btn:focus-visible {
  outline: 2px solid var(--accent-blue);
  outline-offset: 2px;
}

.btn-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

/* File Preview */
.file-preview {
  text-align: center;
  padding: var(--space-32);
  border: 2px solid var(--accent-blue);
  border-radius: var(--radius-lg);
  background: rgba(59, 130, 246, 0.05);
  width: 100%;
}

.preview-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-24);
}

.file-info {
  display: flex;
  align-items: center;
  gap: var(--space-16);
  padding: var(--space-20);
  background: var(--card-bg-hover);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-color);
}

.file-icon {
  width: 32px;
  height: 32px;
  color: var(--accent-blue);
  flex-shrink: 0;
}

.file-details {
  text-align: left;
}

.file-name {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: var(--space-4);
}

.file-size {
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.preview-actions {
  display: flex;
  gap: var(--space-16);
  flex-wrap: wrap;
  justify-content: center;
}

/* Upload Progress */
.upload-progress {
  text-align: center;
  padding: var(--space-48);
  width: 100%;
}

.progress-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-24);
}

.progress-icon {
  width: 48px;
  height: 48px;
  color: var(--accent-blue);
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.progress-title {
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--text-primary);
}

.progress-bar-container {
  display: flex;
  align-items: center;
  gap: var(--space-16);
  width: 100%;
  max-width: 300px;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: var(--border-color);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-light));
  border-radius: 4px;
  transition: width 0.3s ease;
  position: relative;
}

.progress-fill::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
  animation: shimmer 1.5s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.progress-percentage {
  font-size: 1rem;
  font-weight: 600;
  color: var(--accent-blue);
  min-width: 40px;
}

/* Success State */
.upload-success {
  text-align: center;
  padding: var(--space-48);
  width: 100%;
}

.success-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-24);
}

.success-icon {
  width: 64px;
  height: 64px;
  color: var(--success-color);
  animation: successPulse 0.6s ease-out;
}

@keyframes successPulse {
  0% { transform: scale(0); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}

.success-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--success-color);
}

.success-message {
  font-size: 1rem;
  color: var(--text-secondary);
  max-width: 400px;
}

/* Error State */
.upload-error {
  text-align: center;
  padding: var(--space-48);
  width: 100%;
}

.error-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-24);
}

.error-icon {
  width: 64px;
  height: 64px;
  color: var(--danger-color);
  animation: errorShake 0.6s ease-out;
}

@keyframes errorShake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-5px); }
  75% { transform: translateX(5px); }
}

.error-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--danger-color);
}

.error-message {
  font-size: 1rem;
  color: var(--text-secondary);
  max-width: 400px;
}

/* Security Note */
.security-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-12);
  margin-top: var(--space-32);
  padding: var(--space-16);
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: var(--radius-md);
  color: var(--success-color);
}

.security-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.security-note p {
  margin: 0;
  font-size: 0.875rem;
  font-weight: 500;
}

/* Utility Classes */
.hidden {
  display: none !important;
}

/* Animations */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes fadeInDown {
  from {
    opacity: 0;
    transform: translateY(-30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Responsive Design */
@media (max-width: 768px) {
  .app-container {
    padding: var(--space-24) var(--space-16);
  }
  
  .logo-text {
    font-size: 2rem;
  }
  
  .upload-card {
    padding: var(--space-32);
  }
  
  .upload-content {
    padding: var(--space-24);
  }
  
  .upload-title {
    font-size: 1.25rem;
  }
  
  .preview-actions {
    flex-direction: column;
  }
  
  .btn {
    width: 100%;
    justify-content: center;
  }
  
  .file-info {
    flex-direction: column;
    text-align: center;
    gap: var(--space-12);
  }
  
  .file-details {
    text-align: center;
  }
}

@media (max-width: 480px) {
  .logo {
    flex-direction: column;
    gap: var(--space-12);
  }
  
  .logo-icon {
    width: 40px;
    height: 40px;
  }
  
  .logo-text {
    font-size: 1.75rem;
  }
  
  .upload-card {
    padding: var(--space-24);
  }
  
  .upload-content {
    padding: var(--space-16);
  }
  
  .select-file-btn {
    padding: var(--space-12) var(--space-24);
    font-size: 1rem;
  }
}

/* Remove any unwanted pseudo-elements or focus outlines that could cause visual artifacts */
*::before,
*::after {
  pointer-events: none;
}

/* Ensure no stray elements appear */
body::before,
body::after,
html::before,
html::after {
  display: none;
}
  </style>
</head>
<body>
  <div class="app-container">
    <!-- Logo Section -->
    <header class="logo-section">
      <div class="logo">
        <i data-lucide="brain" class="logo-icon"></i>
        <h1 class="logo-text">JAAM AI</h1>
      </div>
      <p class="logo-subtitle">MRI Analysis Upload</p>
    </header>

<!-- Upload Section -->
<main class="upload-section">
  <div class="upload-card">
    <!-- Upload Area -->
    <form class="upload-area" id="uploadArea" action="/upload" method="POST" enctype="multipart/form-data">
      <div class="upload-content" id="uploadContent">
        <i data-lucide="upload-cloud" class="upload-icon"></i>
        <h2 class="upload-title">Drag and drop your MRI scan here</h2>
        <p class="upload-subtitle">or click to select file</p>
        <button type="button" class="select-file-btn" id="selectFileBtn">
          <i data-lucide="folder-open" class="btn-icon"></i>
          Choose File
        </button>
        <p class="upload-info">Supports JPEG, PNG, DICOM files up to 50MB</p>
      </div>

      <!-- File input -->
      <input type="file" id="fileInput" name="file" accept=".jpg,.jpeg,.png,.dicom,.dcm" style="display: none;">

      <!-- Upload button -->
      <button type="submit" class="btn btn-primary" id="uploadBtn">
        <i data-lucide="upload" class="btn-icon"></i>
        Upload Scan
      </button>
    </form>


<!-- File Preview -->
<div class="file-preview hidden" id="filePreview">
  <div class="preview-content">
    <div class="file-info">
      <i data-lucide="file" class="file-icon"></i>
      <div class="file-details">
        <h3 class="file-name" id="fileName"></h3>
        <p class="file-size" id="fileSize"></p>
      </div>
    </div>
    
    <!-- Image preview -->
    <div class="image-preview mt-2">
      <img id="previewImage" src="" alt="Preview" style="max-width: 100%; display: none; border-radius: 8px;">
    </div>

    <div class="preview-actions">
      <button class="btn btn-primary" id="uploadBtn">
        <i data-lucide="upload" class="btn-icon"></i>
        Upload Scan
      </button>
      <button class="btn btn-secondary" id="cancelBtn">
        <i data-lucide="x" class="btn-icon"></i>
        Cancel
      </button>
    </div>
  </div>
</div>


          <!-- Upload Progress -->
          <div class="upload-progress hidden" id="uploadProgress">
            <div class="progress-content">
              <i data-lucide="loader" class="progress-icon spinning"></i>
              <h3 class="progress-title">Uploading your MRI scan...</h3>
              <div class="progress-bar-container">
                <div class="progress-bar">
                  <div class="progress-fill" id="progressFill"></div>
                </div>
                <span class="progress-percentage" id="progressPercentage">0%</span>
              </div>
              <button class="btn btn-secondary btn-small" id="cancelUploadBtn">
                <i data-lucide="x" class="btn-icon"></i>
                Cancel Upload
              </button>
            </div>
          </div>

          <!-- Success -->
          <div class="upload-success hidden" id="uploadSuccess">
            <div class="success-content">
              <i data-lucide="check-circle" class="success-icon"></i>
              <h3 class="success-title">Upload Complete!</h3>
              <p class="success-message">Your MRI scan has been successfully uploaded and is being processed.</p>
              <button class="btn btn-primary" id="uploadAnotherBtn">
                <i data-lucide="plus" class="btn-icon"></i>
                Upload Another Scan
              </button>
            </div>
          </div>

          <!-- Error -->
          <div class="upload-error hidden" id="uploadError">
            <div class="error-content">
              <i data-lucide="alert-circle" class="error-icon"></i>
              <h3 class="error-title">Upload Failed</h3>
              <p class="error-message" id="errorMessage">There was an error uploading your file. Please try again.</p>
              <button class="btn btn-primary" id="retryBtn">
                <i data-lucide="refresh-cw" class="btn-icon"></i>
                Try Again
              </button>
            </div>
          </div>
        </div>

        <!-- Security Note -->
        <div class="security-note">
          <i data-lucide="shield-check" class="security-icon"></i>
          <p>Secure, encrypted upload for medical data</p>
        </div>
      </div>
    </main>
  </div>

  <!-- Hidden file input -->
  <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.dicom,.dcm" multiple="false" style="display: none;">

  <!-- ✅ Inline JavaScript -->
  <script>
    // Initialize Lucide icons and app functionality
document.addEventListener('DOMContentLoaded', function() {
    lucide.createIcons();
    initializeUploadFunctionality();
});

// Global variables
let selectedFile = null;
let uploadInProgress = false;

// File upload settings from application data
const uploadSettings = {
    maxFileSize: 50 * 1024 * 1024, // 50MB in bytes
    allowedFormats: ['image/jpeg', 'image/jpg', 'image/png', 'application/dicom'],
    allowedExtensions: ['.jpg', '.jpeg', '.png', '.dicom', '.dcm']
};

// DOM elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    uploadContent: document.getElementById('uploadContent'),
    filePreview: document.getElementById('filePreview'),
    uploadProgress: document.getElementById('uploadProgress'),
    uploadSuccess: document.getElementById('uploadSuccess'),
    uploadError: document.getElementById('uploadError'),
    selectFileBtn: document.getElementById('selectFileBtn'),
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    cancelBtn: document.getElementById('cancelBtn'),
    cancelUploadBtn: document.getElementById('cancelUploadBtn'),
    uploadAnotherBtn: document.getElementById('uploadAnotherBtn'),
    retryBtn: document.getElementById('retryBtn'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    progressFill: document.getElementById('progressFill'),
    progressPercentage: document.getElementById('progressPercentage'),
    errorMessage: document.getElementById('errorMessage')
};

function initializeUploadFunctionality() {
    // File input change handler
    elements.fileInput.addEventListener('change', handleFileSelection);
    
    // Select file button click handler
    elements.selectFileBtn.addEventListener('click', () => {
        if (!uploadInProgress) {
            elements.fileInput.click();
        }
    });
    
    // Upload area click handler
    elements.uploadContent.addEventListener('click', () => {
        if (!uploadInProgress) {
            elements.fileInput.click();
        }
    });
    
    // Drag and drop handlers
    elements.uploadContent.addEventListener('dragover', handleDragOver);
    elements.uploadContent.addEventListener('dragenter', handleDragEnter);
    elements.uploadContent.addEventListener('dragleave', handleDragLeave);
    elements.uploadContent.addEventListener('drop', handleDrop);
    
    // Button event handlers
    elements.uploadBtn.addEventListener('click', startUpload);
    elements.cancelBtn.addEventListener('click', cancelFileSelection);
    elements.cancelUploadBtn.addEventListener('click', cancelUpload);
    elements.uploadAnotherBtn.addEventListener('click', resetUpload);
    elements.retryBtn.addEventListener('click', retryUpload);
    
    // Prevent default drag behaviors on document
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => e.preventDefault());
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    if (!uploadInProgress) {
        elements.uploadContent.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    // Only remove drag-over if we're leaving the upload content area
    if (!elements.uploadContent.contains(e.relatedTarget)) {
        elements.uploadContent.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadContent.classList.remove('drag-over');
    
    if (uploadInProgress) return;
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// File selection handler
function handleFileSelection(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

// Process selected file
function processFile(file) {
    // Validate file
    const validation = validateFile(file);
    if (!validation.isValid) {
        showError(validation.error);
        return;
    }
    
    selectedFile = file;
    showFilePreview(file);
}

// File validation
function validateFile(file) {
    // Check file size
    if (file.size > uploadSettings.maxFileSize) {
        return {
            isValid: false,
            error: `File size exceeds the maximum limit of 50MB. Your file is ${formatFileSize(file.size)}.`
        };
    }
    
    // Check file type
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    const isValidExtension = uploadSettings.allowedExtensions.includes(fileExtension);
    const isValidMimeType = uploadSettings.allowedFormats.includes(file.type);
    
    if (!isValidExtension && !isValidMimeType) {
        return {
            isValid: false,
            error: 'Invalid file format. Please upload a JPEG, PNG, or DICOM file.'
        };
    }
    
    return { isValid: true };
}

// Show file preview
function showFilePreview(file) {
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);

    // === Add image preview ===
    let previewImage = document.getElementById('previewImage');
    if (!previewImage) {
        // create img element if it doesn't exist
        previewImage = document.createElement('img');
        previewImage.id = 'previewImage';
        previewImage.style.maxWidth = '100%';
        previewImage.style.borderRadius = '8px';
        previewImage.style.display = 'none'; // initially hidden
        // append inside filePreview content
        elements.filePreview.querySelector('.preview-content').insertBefore(previewImage, elements.filePreview.querySelector('.preview-actions'));
    }

    // Only show for image files
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        previewImage.style.display = 'none';
    }
    // === End image preview ===

    showState('preview');
}


// Start upload process
function startUpload() {
    if (!selectedFile || uploadInProgress) return;
    
    uploadInProgress = true;
    showState('progress');
    simulateUpload();
}

// Simulate upload progress
function simulateUpload() {
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        
        if (progress >= 100) {
            progress = 100;
            clearInterval(progressInterval);
            completeUpload();
        }
        
        updateProgress(progress);
    }, 200);
    
    // Store interval ID for potential cancellation
    window.uploadInterval = progressInterval;
}

// Update progress display
function updateProgress(progress) {
    const roundedProgress = Math.round(progress);
    elements.progressFill.style.width = `${roundedProgress}%`;
    elements.progressPercentage.textContent = `${roundedProgress}%`;
}

// Complete upload
function completeUpload() {
    uploadInProgress = false;
    
    // Simulate random success/failure for demo purposes
    // In real implementation, this would be based on actual upload response
    const isSuccess = Math.random() > 0.1; // 90% success rate for demo
    
    if (isSuccess) {
        showState('success');
    } else {
        showError('Upload failed due to a network error. Please check your connection and try again.');
    }
}

// Cancel file selection
function cancelFileSelection() {
    selectedFile = null;
    elements.fileInput.value = '';
    showState('initial');
}

// Cancel upload in progress
function cancelUpload() {
    if (window.uploadInterval) {
        clearInterval(window.uploadInterval);
        window.uploadInterval = null;
    }
    
    uploadInProgress = false;
    showState('preview');
}

// Reset for new upload
function resetUpload() {
    selectedFile = null;
    elements.fileInput.value = '';
    uploadInProgress = false;
    showState('initial');
}

// Retry failed upload
function retryUpload() {
    if (selectedFile) {
        startUpload();
    } else {
        resetUpload();
    }
}

// Show error state
function showError(message) {
    elements.errorMessage.textContent = message;
    uploadInProgress = false;
    showState('error');
}

// State management
function showState(state) {
    // Hide all states
    elements.uploadContent.classList.add('hidden');
    elements.filePreview.classList.add('hidden');
    elements.uploadProgress.classList.add('hidden');
    elements.uploadSuccess.classList.add('hidden');
    elements.uploadError.classList.add('hidden');
    
    // Show selected state
    switch (state) {
        case 'initial':
            elements.uploadContent.classList.remove('hidden');
            break;
        case 'preview':
            elements.filePreview.classList.remove('hidden');
            break;
        case 'progress':
            elements.uploadProgress.classList.remove('hidden');
            updateProgress(0);
            break;
        case 'success':
            elements.uploadSuccess.classList.remove('hidden');
            break;
        case 'error':
            elements.uploadError.classList.remove('hidden');
            break;
    }
    
    // Reinitialize icons for the new state
    setTimeout(() => {
        lucide.createIcons();
    }, 50);
}

// Utility function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Enhanced accessibility and keyboard navigation
document.addEventListener('keydown', function(e) {
    // Handle Escape key to cancel current operation
    if (e.key === 'Escape') {
        if (uploadInProgress) {
            cancelUpload();
        } else if (selectedFile) {
            cancelFileSelection();
        }
    }
    
    // Handle Enter/Space on focusable elements
    if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('upload-content')) {
        e.preventDefault();
        if (!uploadInProgress) {
            elements.fileInput.click();
        }
    }
});

// Add focus management for better accessibility
function addFocusManagement() {
    const focusableElements = document.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid var(--accent-blue)';
            this.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = '';
            this.style.outlineOffset = '';
        });
    });
}

// Initialize focus management when DOM is ready
document.addEventListener('DOMContentLoaded', addFocusManagement);

// Add visual feedback for button interactions
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('.btn, .select-file-btn');
    
    buttons.forEach(button => {
        button.addEventListener('mousedown', function() {
            this.style.transform = 'scale(0.98)';
        });
        
        button.addEventListener('mouseup', function() {
            this.style.transform = '';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = '';
        });
    });
});

// Performance optimization: Lazy load heavy animations
function initLazyAnimations() {
    const animatedElements = document.querySelectorAll('.upload-icon, .logo-icon');
    
    const animationObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                animationObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.3
    });
    
    animatedElements.forEach(el => {
        el.style.animationPlayState = 'paused';
        animationObserver.observe(el);
    });
}

// Initialize lazy animations after a delay
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(initLazyAnimations, 500);
});

// Handle window resize and orientation changes
window.addEventListener('resize', function() {
    // Ensure proper layout on mobile orientation changes
    setTimeout(() => {
        lucide.createIcons();
    }, 100);
});

// Add touch support for mobile devices
document.addEventListener('DOMContentLoaded', function() {
    // Enhanced touch support for upload area
    elements.uploadContent.addEventListener('touchstart', function() {
        if (!uploadInProgress) {
            this.style.transform = 'scale(0.98)';
        }
    });
    
    elements.uploadContent.addEventListener('touchend', function() {
        this.style.transform = '';
    });
    
    elements.uploadContent.addEventListener('touchcancel', function() {
        this.style.transform = '';
    });
});

// Error handling for file operations
window.addEventListener('error', function(e) {
    console.error('Upload application error:', e.error);
    if (uploadInProgress) {
        showError('An unexpected error occurred. Please refresh the page and try again.');
    }
});

// Cleanup function for when page is unloaded
window.addEventListener('beforeunload', function() {
    if (window.uploadInterval) {
        clearInterval(window.uploadInterval);
    }
});

// Add progress animation enhancement
function enhanceProgressAnimation() {
    const progressFill = elements.progressFill;
    if (progressFill) {
        // Add shimmer effect classes dynamically
        progressFill.classList.add('progress-shimmer');
    }
}

// Initialize enhancements
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(enhanceProgressAnimation, 1000);
});
  </script>
</body>
</html>

'''

RESULT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analysis Results - JAAM AI</title>
  <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>

  <!-- ✅ Inline CSS -->
  <style>
    :root {
  /* Primitive Color Tokens */
  --color-white: rgba(255, 255, 255, 1);
  --color-black: rgba(0, 0, 0, 1);
  --color-cream-50: rgba(252, 252, 249, 1);
  --color-cream-100: rgba(255, 255, 253, 1);
  --color-gray-200: rgba(245, 245, 245, 1);
  --color-gray-300: rgba(167, 169, 169, 1);
  --color-gray-400: rgba(119, 124, 124, 1);
  --color-slate-500: rgba(98, 108, 113, 1);
  --color-brown-600: rgba(94, 82, 64, 1);
  --color-charcoal-700: rgba(31, 33, 33, 1);
  --color-charcoal-800: rgba(38, 40, 40, 1);
  --color-slate-900: rgba(19, 52, 59, 1);
  --color-teal-300: rgba(50, 184, 198, 1);
  --color-teal-400: rgba(45, 166, 178, 1);
  --color-teal-500: rgba(33, 128, 141, 1);
  --color-teal-600: rgba(29, 116, 128, 1);
  --color-teal-700: rgba(26, 104, 115, 1);
  --color-teal-800: rgba(41, 150, 161, 1);
  --color-red-400: rgba(255, 84, 89, 1);
  --color-red-500: rgba(192, 21, 47, 1);
  --color-orange-400: rgba(230, 129, 97, 1);
  --color-orange-500: rgba(168, 75, 47, 1);

  /* RGB versions for opacity control */
  --color-brown-600-rgb: 94, 82, 64;
  --color-teal-500-rgb: 33, 128, 141;
  --color-slate-900-rgb: 19, 52, 59;
  --color-slate-500-rgb: 98, 108, 113;
  --color-red-500-rgb: 192, 21, 47;
  --color-red-400-rgb: 255, 84, 89;
  --color-orange-500-rgb: 168, 75, 47;
  --color-orange-400-rgb: 230, 129, 97;

  /* Background color tokens (Light Mode) */
  --color-bg-1: rgba(59, 130, 246, 0.08); /* Light blue */
  --color-bg-2: rgba(245, 158, 11, 0.08); /* Light yellow */
  --color-bg-3: rgba(34, 197, 94, 0.08); /* Light green */
  --color-bg-4: rgba(239, 68, 68, 0.08); /* Light red */
  --color-bg-5: rgba(147, 51, 234, 0.08); /* Light purple */
  --color-bg-6: rgba(249, 115, 22, 0.08); /* Light orange */
  --color-bg-7: rgba(236, 72, 153, 0.08); /* Light pink */
  --color-bg-8: rgba(6, 182, 212, 0.08); /* Light cyan */

  /* Semantic Color Tokens (Light Mode) */
  --color-background: var(--color-cream-50);
  --color-surface: var(--color-cream-100);
  --color-text: var(--color-slate-900);
  --color-text-secondary: var(--color-slate-500);
  --color-primary: var(--color-teal-500);
  --color-primary-hover: var(--color-teal-600);
  --color-primary-active: var(--color-teal-700);
  --color-secondary: rgba(var(--color-brown-600-rgb), 0.12);
  --color-secondary-hover: rgba(var(--color-brown-600-rgb), 0.2);
  --color-secondary-active: rgba(var(--color-brown-600-rgb), 0.25);
  --color-border: rgba(var(--color-brown-600-rgb), 0.2);
  --color-btn-primary-text: var(--color-cream-50);
  --color-card-border: rgba(var(--color-brown-600-rgb), 0.12);
  --color-card-border-inner: rgba(var(--color-brown-600-rgb), 0.12);
  --color-error: var(--color-red-500);
  --color-success: var(--color-teal-500);
  --color-warning: var(--color-orange-500);
  --color-info: var(--color-slate-500);
  --color-focus-ring: rgba(var(--color-teal-500-rgb), 0.4);
  --color-select-caret: rgba(var(--color-slate-900-rgb), 0.8);

  /* Common style patterns */
  --focus-ring: 0 0 0 3px var(--color-focus-ring);
  --focus-outline: 2px solid var(--color-primary);
  --status-bg-opacity: 0.15;
  --status-border-opacity: 0.25;
  --select-caret-light: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23134252' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  --select-caret-dark: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f5f5f5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");

  /* RGB versions for opacity control */
  --color-success-rgb: 33, 128, 141;
  --color-error-rgb: 192, 21, 47;
  --color-warning-rgb: 168, 75, 47;
  --color-info-rgb: 98, 108, 113;

  /* Typography */
  --font-family-base: "FKGroteskNeue", "Geist", "Inter", -apple-system,
    BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --font-family-mono: "Berkeley Mono", ui-monospace, SFMono-Regular, Menlo,
    Monaco, Consolas, monospace;
  --font-size-xs: 11px;
  --font-size-sm: 12px;
  --font-size-base: 14px;
  --font-size-md: 14px;
  --font-size-lg: 16px;
  --font-size-xl: 18px;
  --font-size-2xl: 20px;
  --font-size-3xl: 24px;
  --font-size-4xl: 30px;
  --font-weight-normal: 400;
  --font-weight-medium: 500;
  --font-weight-semibold: 550;
  --font-weight-bold: 600;
  --line-height-tight: 1.2;
  --line-height-normal: 1.5;
  --letter-spacing-tight: -0.01em;

  /* Spacing */
  --space-0: 0;
  --space-1: 1px;
  --space-2: 2px;
  --space-4: 4px;
  --space-6: 6px;
  --space-8: 8px;
  --space-10: 10px;
  --space-12: 12px;
  --space-16: 16px;
  --space-20: 20px;
  --space-24: 24px;
  --space-32: 32px;

  /* Border Radius */
  --radius-sm: 6px;
  --radius-base: 8px;
  --radius-md: 10px;
  --radius-lg: 12px;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.02);
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.04),
    0 2px 4px -1px rgba(0, 0, 0, 0.02);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.04),
    0 4px 6px -2px rgba(0, 0, 0, 0.02);
  --shadow-inset-sm: inset 0 1px 0 rgba(255, 255, 255, 0.15),
    inset 0 -1px 0 rgba(0, 0, 0, 0.03);

  /* Animation */
  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --ease-standard: cubic-bezier(0.16, 1, 0.3, 1);

  /* Layout */
  --container-sm: 640px;
  --container-md: 768px;
  --container-lg: 1024px;
  --container-xl: 1280px;
}

/* Dark mode colors */
@media (prefers-color-scheme: dark) {
  :root {
    /* RGB versions for opacity control (Dark Mode) */
    --color-gray-400-rgb: 119, 124, 124;
    --color-teal-300-rgb: 50, 184, 198;
    --color-gray-300-rgb: 167, 169, 169;
    --color-gray-200-rgb: 245, 245, 245;

    /* Background color tokens (Dark Mode) */
    --color-bg-1: rgba(29, 78, 216, 0.15); /* Dark blue */
    --color-bg-2: rgba(180, 83, 9, 0.15); /* Dark yellow */
    --color-bg-3: rgba(21, 128, 61, 0.15); /* Dark green */
    --color-bg-4: rgba(185, 28, 28, 0.15); /* Dark red */
    --color-bg-5: rgba(107, 33, 168, 0.15); /* Dark purple */
    --color-bg-6: rgba(194, 65, 12, 0.15); /* Dark orange */
    --color-bg-7: rgba(190, 24, 93, 0.15); /* Dark pink */
    --color-bg-8: rgba(8, 145, 178, 0.15); /* Dark cyan */

    /* Semantic Color Tokens (Dark Mode) */
    --color-background: var(--color-charcoal-700);
    --color-surface: var(--color-charcoal-800);
    --color-text: var(--color-gray-200);
    --color-text-secondary: rgba(var(--color-gray-300-rgb), 0.7);
    --color-primary: var(--color-teal-300);
    --color-primary-hover: var(--color-teal-400);
    --color-primary-active: var(--color-teal-800);
    --color-secondary: rgba(var(--color-gray-400-rgb), 0.15);
    --color-secondary-hover: rgba(var(--color-gray-400-rgb), 0.25);
    --color-secondary-active: rgba(var(--color-gray-400-rgb), 0.3);
    --color-border: rgba(var(--color-gray-400-rgb), 0.3);
    --color-error: var(--color-red-400);
    --color-success: var(--color-teal-300);
    --color-warning: var(--color-orange-400);
    --color-info: var(--color-gray-300);
    --color-focus-ring: rgba(var(--color-teal-300-rgb), 0.4);
    --color-btn-primary-text: var(--color-slate-900);
    --color-card-border: rgba(var(--color-gray-400-rgb), 0.2);
    --color-card-border-inner: rgba(var(--color-gray-400-rgb), 0.15);
    --shadow-inset-sm: inset 0 1px 0 rgba(255, 255, 255, 0.1),
      inset 0 -1px 0 rgba(0, 0, 0, 0.15);
    --button-border-secondary: rgba(var(--color-gray-400-rgb), 0.2);
    --color-border-secondary: rgba(var(--color-gray-400-rgb), 0.2);
    --color-select-caret: rgba(var(--color-gray-200-rgb), 0.8);

    /* Common style patterns - updated for dark mode */
    --focus-ring: 0 0 0 3px var(--color-focus-ring);
    --focus-outline: 2px solid var(--color-primary);
    --status-bg-opacity: 0.15;
    --status-border-opacity: 0.25;
    --select-caret-light: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23134252' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
    --select-caret-dark: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f5f5f5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");

    /* RGB versions for dark mode */
    --color-success-rgb: var(--color-teal-300-rgb);
    --color-error-rgb: var(--color-red-400-rgb);
    --color-warning-rgb: var(--color-orange-400-rgb);
    --color-info-rgb: var(--color-gray-300-rgb);
  }
}

/* Data attribute for manual theme switching */
[data-color-scheme="dark"] {
  /* RGB versions for opacity control (dark mode) */
  --color-gray-400-rgb: 119, 124, 124;
  --color-teal-300-rgb: 50, 184, 198;
  --color-gray-300-rgb: 167, 169, 169;
  --color-gray-200-rgb: 245, 245, 245;

  /* Colorful background palette - Dark Mode */
  --color-bg-1: rgba(29, 78, 216, 0.15); /* Dark blue */
  --color-bg-2: rgba(180, 83, 9, 0.15); /* Dark yellow */
  --color-bg-3: rgba(21, 128, 61, 0.15); /* Dark green */
  --color-bg-4: rgba(185, 28, 28, 0.15); /* Dark red */
  --color-bg-5: rgba(107, 33, 168, 0.15); /* Dark purple */
  --color-bg-6: rgba(194, 65, 12, 0.15); /* Dark orange */
  --color-bg-7: rgba(190, 24, 93, 0.15); /* Dark pink */
  --color-bg-8: rgba(8, 145, 178, 0.15); /* Dark cyan */

  /* Semantic Color Tokens (Dark Mode) */
  --color-background: var(--color-charcoal-700);
  --color-surface: var(--color-charcoal-800);
  --color-text: var(--color-gray-200);
  --color-text-secondary: rgba(var(--color-gray-300-rgb), 0.7);
  --color-primary: var(--color-teal-300);
  --color-primary-hover: var(--color-teal-400);
  --color-primary-active: var(--color-teal-800);
  --color-secondary: rgba(var(--color-gray-400-rgb), 0.15);
  --color-secondary-hover: rgba(var(--color-gray-400-rgb), 0.25);
  --color-secondary-active: rgba(var(--color-gray-400-rgb), 0.3);
  --color-border: rgba(var(--color-gray-400-rgb), 0.3);
  --color-error: var(--color-red-400);
  --color-success: var(--color-teal-300);
  --color-warning: var(--color-orange-400);
  --color-info: var(--color-gray-300);
  --color-focus-ring: rgba(var(--color-teal-300-rgb), 0.4);
  --color-btn-primary-text: var(--color-slate-900);
  --color-card-border: rgba(var(--color-gray-400-rgb), 0.15);
  --color-card-border-inner: rgba(var(--color-gray-400-rgb), 0.15);
  --shadow-inset-sm: inset 0 1px 0 rgba(255, 255, 255, 0.1),
    inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  --color-border-secondary: rgba(var(--color-gray-400-rgb), 0.2);
  --color-select-caret: rgba(var(--color-gray-200-rgb), 0.8);

  /* Common style patterns - updated for dark mode */
  --focus-ring: 0 0 0 3px var(--color-focus-ring);
  --focus-outline: 2px solid var(--color-primary);
  --status-bg-opacity: 0.15;
  --status-border-opacity: 0.25;
  --select-caret-light: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23134252' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
  --select-caret-dark: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23f5f5f5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");

  /* RGB versions for dark mode */
  --color-success-rgb: var(--color-teal-300-rgb);
  --color-error-rgb: var(--color-red-400-rgb);
  --color-warning-rgb: var(--color-orange-400-rgb);
  --color-info-rgb: var(--color-gray-300-rgb);
}

[data-color-scheme="light"] {
  /* RGB versions for opacity control (light mode) */
  --color-brown-600-rgb: 94, 82, 64;
  --color-teal-500-rgb: 33, 128, 141;
  --color-slate-900-rgb: 19, 52, 59;

  /* Semantic Color Tokens (Light Mode) */
  --color-background: var(--color-cream-50);
  --color-surface: var(--color-cream-100);
  --color-text: var(--color-slate-900);
  --color-text-secondary: var(--color-slate-500);
  --color-primary: var(--color-teal-500);
  --color-primary-hover: var(--color-teal-600);
  --color-primary-active: var(--color-teal-700);
  --color-secondary: rgba(var(--color-brown-600-rgb), 0.12);
  --color-secondary-hover: rgba(var(--color-brown-600-rgb), 0.2);
  --color-secondary-active: rgba(var(--color-brown-600-rgb), 0.25);
  --color-border: rgba(var(--color-brown-600-rgb), 0.2);
  --color-btn-primary-text: var(--color-cream-50);
  --color-card-border: rgba(var(--color-brown-600-rgb), 0.12);
  --color-card-border-inner: rgba(var(--color-brown-600-rgb), 0.12);
  --color-error: var(--color-red-500);
  --color-success: var(--color-teal-500);
  --color-warning: var(--color-orange-500);
  --color-info: var(--color-slate-500);
  --color-focus-ring: rgba(var(--color-teal-500-rgb), 0.4);

  /* RGB versions for light mode */
  --color-success-rgb: var(--color-teal-500-rgb);
  --color-error-rgb: var(--color-red-500-rgb);
  --color-warning-rgb: var(--color-orange-500-rgb);
  --color-info-rgb: var(--color-slate-500-rgb);
}

/* Base styles */
html {
  font-size: var(--font-size-base);
  font-family: var(--font-family-base);
  line-height: var(--line-height-normal);
  color: var(--color-text);
  background-color: var(--color-background);
  -webkit-font-smoothing: antialiased;
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
}

*,
*::before,
*::after {
  box-sizing: inherit;
}

/* Typography */
h1,
h2,
h3,
h4,
h5,
h6 {
  margin: 0;
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-tight);
  color: var(--color-text);
  letter-spacing: var(--letter-spacing-tight);
}

h1 {
  font-size: var(--font-size-4xl);
}
h2 {
  font-size: var(--font-size-3xl);
}
h3 {
  font-size: var(--font-size-2xl);
}
h4 {
  font-size: var(--font-size-xl);
}
h5 {
  font-size: var(--font-size-lg);
}
h6 {
  font-size: var(--font-size-md);
}

p {
  margin: 0 0 var(--space-16) 0;
}

a {
  color: var(--color-primary);
  text-decoration: none;
  transition: color var(--duration-fast) var(--ease-standard);
}

a:hover {
  color: var(--color-primary-hover);
}

code,
pre {
  font-family: var(--font-family-mono);
  font-size: calc(var(--font-size-base) * 0.95);
  background-color: var(--color-secondary);
  border-radius: var(--radius-sm);
}

code {
  padding: var(--space-1) var(--space-4);
}

pre {
  padding: var(--space-16);
  margin: var(--space-16) 0;
  overflow: auto;
  border: 1px solid var(--color-border);
}

pre code {
  background: none;
  padding: 0;
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-8) var(--space-16);
  border-radius: var(--radius-base);
  font-size: var(--font-size-base);
  font-weight: 500;
  line-height: 1.5;
  cursor: pointer;
  transition: all var(--duration-normal) var(--ease-standard);
  border: none;
  text-decoration: none;
  position: relative;
}

.btn:focus-visible {
  outline: none;
  box-shadow: var(--focus-ring);
}

.btn--primary {
  background: var(--color-primary);
  color: var(--color-btn-primary-text);
}

.btn--primary:hover {
  background: var(--color-primary-hover);
}

.btn--primary:active {
  background: var(--color-primary-active);
}

.btn--secondary {
  background: var(--color-secondary);
  color: var(--color-text);
}

.btn--secondary:hover {
  background: var(--color-secondary-hover);
}

.btn--secondary:active {
  background: var(--color-secondary-active);
}

.btn--outline {
  background: transparent;
  border: 1px solid var(--color-border);
  color: var(--color-text);
}

.btn--outline:hover {
  background: var(--color-secondary);
}

.btn--sm {
  padding: var(--space-4) var(--space-12);
  font-size: var(--font-size-sm);
  border-radius: var(--radius-sm);
}

.btn--lg {
  padding: var(--space-10) var(--space-20);
  font-size: var(--font-size-lg);
  border-radius: var(--radius-md);
}

.btn--full-width {
  width: 100%;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Form elements */
.form-control {
  display: block;
  width: 100%;
  padding: var(--space-8) var(--space-12);
  font-size: var(--font-size-md);
  line-height: 1.5;
  color: var(--color-text);
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-base);
  transition: border-color var(--duration-fast) var(--ease-standard),
    box-shadow var(--duration-fast) var(--ease-standard);
}

textarea.form-control {
  font-family: var(--font-family-base);
  font-size: var(--font-size-base);
}

select.form-control {
  padding: var(--space-8) var(--space-12);
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  background-image: var(--select-caret-light);
  background-repeat: no-repeat;
  background-position: right var(--space-12) center;
  background-size: 16px;
  padding-right: var(--space-32);
}

/* Add a dark mode specific caret */
@media (prefers-color-scheme: dark) {
  select.form-control {
    background-image: var(--select-caret-dark);
  }
}

/* Also handle data-color-scheme */
[data-color-scheme="dark"] select.form-control {
  background-image: var(--select-caret-dark);
}

[data-color-scheme="light"] select.form-control {
  background-image: var(--select-caret-light);
}

.form-control:focus {
  border-color: var(--color-primary);
  outline: var(--focus-outline);
}

.form-label {
  display: block;
  margin-bottom: var(--space-8);
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
}

.form-group {
  margin-bottom: var(--space-16);
}

/* Card component */
.card {
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--color-card-border);
  box-shadow: var(--shadow-sm);
  overflow: hidden;
  transition: box-shadow var(--duration-normal) var(--ease-standard);
}

.card:hover {
  box-shadow: var(--shadow-md);
}

.card__body {
  padding: var(--space-16);
}

.card__header,
.card__footer {
  padding: var(--space-16);
  border-bottom: 1px solid var(--color-card-border-inner);
}

/* Status indicators - simplified with CSS variables */
.status {
  display: inline-flex;
  align-items: center;
  padding: var(--space-6) var(--space-12);
  border-radius: var(--radius-full);
  font-weight: var(--font-weight-medium);
  font-size: var(--font-size-sm);
}

.status--success {
  background-color: rgba(
    var(--color-success-rgb, 33, 128, 141),
    var(--status-bg-opacity)
  );
  color: var(--color-success);
  border: 1px solid
    rgba(var(--color-success-rgb, 33, 128, 141), var(--status-border-opacity));
}

.status--error {
  background-color: rgba(
    var(--color-error-rgb, 192, 21, 47),
    var(--status-bg-opacity)
  );
  color: var(--color-error);
  border: 1px solid
    rgba(var(--color-error-rgb, 192, 21, 47), var(--status-border-opacity));
}

.status--warning {
  background-color: rgba(
    var(--color-warning-rgb, 168, 75, 47),
    var(--status-bg-opacity)
  );
  color: var(--color-warning);
  border: 1px solid
    rgba(var(--color-warning-rgb, 168, 75, 47), var(--status-border-opacity));
}

.status--info {
  background-color: rgba(
    var(--color-info-rgb, 98, 108, 113),
    var(--status-bg-opacity)
  );
  color: var(--color-info);
  border: 1px solid
    rgba(var(--color-info-rgb, 98, 108, 113), var(--status-border-opacity));
}

/* Container layout */
.container {
  width: 100%;
  margin-right: auto;
  margin-left: auto;
  padding-right: var(--space-16);
  padding-left: var(--space-16);
}

@media (min-width: 640px) {
  .container {
    max-width: var(--container-sm);
  }
}
@media (min-width: 768px) {
  .container {
    max-width: var(--container-md);
  }
}
@media (min-width: 1024px) {
  .container {
    max-width: var(--container-lg);
  }
}
@media (min-width: 1280px) {
  .container {
    max-width: var(--container-xl);
  }
}

/* Utility classes */
.flex {
  display: flex;
}
.flex-col {
  flex-direction: column;
}
.items-center {
  align-items: center;
}
.justify-center {
  justify-content: center;
}
.justify-between {
  justify-content: space-between;
}
.gap-4 {
  gap: var(--space-4);
}
.gap-8 {
  gap: var(--space-8);
}
.gap-16 {
  gap: var(--space-16);
}

.m-0 {
  margin: 0;
}
.mt-8 {
  margin-top: var(--space-8);
}
.mb-8 {
  margin-bottom: var(--space-8);
}
.mx-8 {
  margin-left: var(--space-8);
  margin-right: var(--space-8);
}
.my-8 {
  margin-top: var(--space-8);
  margin-bottom: var(--space-8);
}

.p-0 {
  padding: 0;
}
.py-8 {
  padding-top: var(--space-8);
  padding-bottom: var(--space-8);
}
.px-8 {
  padding-left: var(--space-8);
  padding-right: var(--space-8);
}
.py-16 {
  padding-top: var(--space-16);
  padding-bottom: var(--space-16);
}
.px-16 {
  padding-left: var(--space-16);
  padding-right: var(--space-16);
}

.block {
  display: block;
}
.hidden {
  display: none;
}

/* Accessibility */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

:focus-visible {
  outline: var(--focus-outline);
  outline-offset: 2px;
}

/* Dark mode specifics */
[data-color-scheme="dark"] .btn--outline {
  border: 1px solid var(--color-border-secondary);
}

@font-face {
  font-family: 'FKGroteskNeue';
  src: url('https://r2cdn.perplexity.ai/fonts/FKGroteskNeue.woff2')
    format('woff2');
}

/* END PERPLEXITY DESIGN SYSTEM */
/* Override design system for specific dark theme */
:root {
  --bg-primary: #0a0a0a;
  --bg-secondary: #18181a;
  --border-color: #26262b;
  --text-primary: #ffffff;
  --text-secondary: #a3a3a3;
  --blue-primary: rgba(59, 130, 246, 0.6);
  --blue-secondary: rgba(37, 99, 235, 0.3);
  --success-color: #10b981;
  --warning-color: #f59e0b;
  --danger-color: #ef4444;
  --primary-blue: #3b82f6;
}

/* Base Styles */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: var(--font-family-base);
  background-color: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  min-height: 100vh;
  overflow-x: hidden;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--space-16);
}

/* Animated Background */
.background-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
  z-index: 1;
  overflow: hidden;
  background: radial-gradient(ellipse at center, rgba(59, 130, 246, 0.08) 0%, rgba(37, 99, 235, 0.04) 35%, transparent 70%);
}

.neuron-animation {
  position: relative;
  width: 100%;
  height: 100%;
}

.neuron {
  position: absolute;
  width: 12px;
  height: 12px;
  background: radial-gradient(circle, rgba(59, 130, 246, 0.8) 0%, rgba(59, 130, 246, 0.4) 50%, rgba(59, 130, 246, 0.1) 100%);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.6), 0 0 40px rgba(59, 130, 246, 0.3);
  animation: float 12s ease-in-out infinite;
}

.neuron::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 20px;
  height: 20px;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, transparent 70%);
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

.neuron-1 {
  top: 15%;
  left: 10%;
  animation-delay: 0s;
}

.neuron-2 {
  top: 60%;
  left: 20%;
  animation-delay: -3s;
}

.neuron-3 {
  top: 35%;
  right: 15%;
  animation-delay: -6s;
}

.neuron-4 {
  top: 75%;
  right: 25%;
  animation-delay: -4s;
}

.neuron-5 {
  top: 25%;
  left: 55%;
  animation-delay: -8s;
}

.synapse {
  position: absolute;
  background: linear-gradient(90deg, transparent 0%, rgba(59, 130, 246, 0.6) 30%, rgba(59, 130, 246, 0.8) 50%, rgba(59, 130, 246, 0.6) 70%, transparent 100%);
  height: 3px;
  border-radius: 2px;
  animation: synapseFlow 4s ease-in-out infinite;
  box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
}

.synapse::before {
  content: '';
  position: absolute;
  top: -2px;
  left: 0;
  right: 0;
  bottom: -2px;
  background: inherit;
  opacity: 0.5;
  filter: blur(2px);
}

.synapse-1 {
  top: 20%;
  left: 15%;
  width: 180px;
  transform: rotate(35deg);
  animation-delay: 0s;
}

.synapse-2 {
  top: 45%;
  right: 20%;
  width: 150px;
  transform: rotate(-25deg);
  animation-delay: -1.5s;
}

.synapse-3 {
  bottom: 25%;
  left: 35%;
  width: 200px;
  transform: rotate(55deg);
  animation-delay: -3s;
}

/* Additional neurons for more activity */
.neuron:nth-child(6) {
  top: 45%;
  left: 85%;
  width: 10px;
  height: 10px;
  animation-delay: -10s;
}

.neuron:nth-child(7) {
  top: 80%;
  left: 60%;
  width: 14px;
  height: 14px;
  animation-delay: -5s;
}

.neuron:nth-child(8) {
  top: 10%;
  right: 40%;
  width: 11px;
  height: 11px;
  animation-delay: -7s;
}

@keyframes float {
  0%, 100% {
    transform: translate(0, 0) scale(1);
    opacity: 0.7;
  }
  25% {
    transform: translate(40px, -25px) scale(1.1);
    opacity: 1;
  }
  50% {
    transform: translate(-15px, 35px) scale(0.9);
    opacity: 0.8;
  }
  75% {
    transform: translate(25px, 15px) scale(1.05);
    opacity: 0.9;
  }
}

@keyframes pulse {
  0%, 100% {
    opacity: 0.3;
    transform: translate(-50%, -50%) scale(1);
  }
  50% {
    opacity: 0.7;
    transform: translate(-50%, -50%) scale(1.3);
  }
}

@keyframes synapseFlow {
  0%, 100% {
    opacity: 0.4;
    transform: scaleX(1) scaleY(1);
  }
  25% {
    opacity: 0.8;
    transform: scaleX(1.1) scaleY(1.2);
  }
  50% {
    opacity: 1;
    transform: scaleX(1.2) scaleY(0.8);
  }
  75% {
    opacity: 0.6;
    transform: scaleX(0.9) scaleY(1.1);
  }
}

/* Header */
.header {
  position: relative;
  z-index: 10;
  background: rgba(24, 24, 26, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color);
  padding: var(--space-16) 0;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-16);
}

.back-button {
  display: flex;
  align-items: center;
  gap: var(--space-8);
  background: none;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  padding: var(--space-8) var(--space-16);
  border-radius: var(--radius-base);
  cursor: pointer;
  transition: all 0.2s ease;
}

.back-button:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.brand {
  display: flex;
  align-items: center;
  gap: var(--space-12);
}

.brand i {
  width: var(--space-24);
  height: var(--space-24);
  color: var(--primary-blue);
}

.brand h1 {
  font-size: var(--font-size-2xl);
  font-weight: var(--font-weight-bold);
  color: var(--text-primary);
}

.header-actions {
  display: flex;
  gap: var(--space-12);
}

/* Buttons */
.btn {
  display: inline-flex;
  align-items: center;
  gap: var(--space-8);
  padding: var(--space-8) var(--space-16);
  border-radius: var(--radius-base);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none;
  border: 1px solid transparent;
}

.btn--secondary {
  background: var(--primary-blue);
  color: white;
}

.btn--secondary:hover {
  background: #2563eb;
}

.btn--outline {
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
}

.btn--outline:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

/* Main Content */
.main-content {
  position: relative;
  z-index: 5;
  padding: var(--space-32) 0;
}

/* Cards */
.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  margin-bottom: var(--space-24);
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  opacity: 0;
  transform: translateY(20px);
}

.card.animate-in {
  opacity: 1;
  transform: translateY(0);
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-20);
  border-bottom: 1px solid var(--border-color);
  background: rgba(59, 130, 246, 0.05);
}

.card-header h2 {
  font-size: var(--font-size-xl);
  font-weight: var(--font-weight-semibold);
  color: var(--text-primary);
}

.card-body {
  padding: var(--space-20);
  transition: max-height 0.3s ease, opacity 0.3s ease;
  overflow: hidden;
}

.card-body.collapsed {
  max-height: 0;
  padding: 0 var(--space-20);
  opacity: 0;
}

/* Status and Badges */
.status {
  display: inline-flex;
  align-items: center;
  padding: var(--space-4) var(--space-12);
  border-radius: var(--radius-full);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
}

.status--processing {
  background: rgba(16, 185, 129, 0.1);
  color: var(--success-color);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.confidence-badge {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--bg-primary);
  padding: var(--space-12);
  border-radius: var(--radius-base);
  border: 1px solid var(--primary-blue);
}

.confidence-value {
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-bold);
  color: var(--primary-blue);
}

.confidence-label {
  font-size: var(--font-size-xs);
  color: var(--text-secondary);
}

/* Patient Info */
.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--space-16);
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.info-item .label {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

.info-item .value {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-medium);
  color: var(--text-primary);
}

/* Diagnosis Results */
.diagnosis-result {
  display: grid;
  grid-template-columns: 1fr 2fr;
  gap: var(--space-32);
  align-items: start;
}

.diagnosis-main h3 {
  font-size: var(--font-size-lg);
  color: var(--text-secondary);
  margin-bottom: var(--space-12);
}

.diagnosis-classification {
  display: flex;
  flex-direction: column;
  gap: var(--space-12);
}

.classification-text {
  font-size: var(--font-size-2xl);
  font-weight: var(--font-weight-bold);
  color: var(--text-primary);
}

.risk-indicator {
  display: inline-flex;
  align-items: center;
  padding: var(--space-6) var(--space-12);
  border-radius: var(--radius-full);
  font-size: var(--font-size-sm);
  font-weight: var(--font-weight-medium);
}

.risk--moderate {
  background: rgba(245, 158, 11, 0.1);
  color: var(--warning-color);
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.confidence-breakdown h4 {
  font-size: var(--font-size-base);
  color: var(--text-secondary);
  margin-bottom: var(--space-16);
}

.progress-bars {
  display: flex;
  flex-direction: column;
  gap: var(--space-12);
}

.progress-item {
  opacity: 0.7;
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.progress-item.active {
  opacity: 1;
}

.progress-item.highlighted {
  opacity: 1;
  transform: scale(1.02);
}

.progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--space-4);
  font-size: var(--font-size-sm);
}

.progress-bar {
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: var(--radius-sm);
  transition: width 0.8s ease;
}

/* Brain Analysis */
.gradcam-section {
  margin-bottom: var(--space-24);
}

.gradcam-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  background: rgba(59, 130, 246, 0.05);
  border: 2px dashed var(--border-color);
  border-radius: var(--radius-base);
  color: var(--text-secondary);
}

.gradcam-placeholder i {
  width: 48px;
  height: 48px;
  margin-bottom: var(--space-12);
  color: var(--primary-blue);
}

.gradcam-placeholder p {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-medium);
  margin-bottom: var(--space-4);
}

.placeholder-text {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

.regions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: var(--space-16);
}

.region-item {
  padding: var(--space-16);
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-base);
  border: 1px solid var(--border-color);
  transition: all 0.2s ease;
  cursor: pointer;
}

.region-item:hover {
  transform: translateX(4px);
}

.region-item.selected {
  border-color: var(--primary-blue);
  background: rgba(59, 130, 246, 0.1);
}

.region-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-8);
}

.region-name {
  font-weight: var(--font-weight-medium);
  color: var(--text-primary);
}

.severity {
  padding: var(--space-2) var(--space-8);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
}

.severity--normal {
  background: rgba(16, 185, 129, 0.1);
  color: var(--success-color);
}

.severity--mild {
  background: rgba(245, 158, 11, 0.1);
  color: var(--warning-color);
}

.severity--moderate {
  background: rgba(239, 68, 68, 0.1);
  color: var(--danger-color);
}

.region-bar {
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: var(--radius-sm);
  margin: var(--space-8) 0;
  overflow: hidden;
}

.region-fill {
  height: 100%;
  border-radius: var(--radius-sm);
  transition: width 0.8s ease;
}

.region-percentage {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
}

/* Recommendations */
.recommendations-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-16);
}

.recommendation-item {
  display: flex;
  align-items: flex-start;
  gap: var(--space-16);
  padding: var(--space-16);
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-base);
  border-left: 4px solid transparent;
  transition: transform 0.2s ease;
}

.recommendation-item:hover {
  transform: translateX(8px);
}

.recommendation-item.priority-high {
  border-left-color: var(--danger-color);
  background: rgba(239, 68, 68, 0.05);
}

.recommendation-item.priority-medium {
  border-left-color: var(--warning-color);
  background: rgba(245, 158, 11, 0.05);
}

.recommendation-item.priority-low {
  border-left-color: var(--success-color);
  background: rgba(16, 185, 129, 0.05);
}

.recommendation-item i {
  width: var(--space-20);
  height: var(--space-20);
  color: var(--primary-blue);
  margin-top: var(--space-2);
}

.recommendation-content h4 {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--text-primary);
  margin-bottom: var(--space-4);
}

.recommendation-content p {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  margin: 0;
}

/* Technical Details */
.tech-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: var(--space-20);
}

.tech-item {
  padding: var(--space-16);
  background: rgba(255, 255, 255, 0.02);
  border-radius: var(--radius-base);
  border: 1px solid var(--border-color);
}

.tech-item h4 {
  font-size: var(--font-size-base);
  font-weight: var(--font-weight-semibold);
  color: var(--text-primary);
  margin-bottom: var(--space-8);
}

.tech-item p {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  margin: 0;
  line-height: 1.5;
}

/* Expand Button */
.expand-button {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  padding: var(--space-4);
  border-radius: var(--radius-sm);
  transition: all 0.2s ease;
}

.expand-button:hover {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.expand-button.expanded i {
  transform: rotate(180deg);
}

.expand-button i {
  width: var(--space-20);
  height: var(--space-20);
  transition: transform 0.2s ease;
}

/* Notifications */
.notification {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  max-width: 400px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-base);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
  transform: translateX(calc(100% + 20px));
  transition: transform 0.3s ease;
}

.notification--show {
  transform: translateX(0);
}

.notification--success {
  border-left: 4px solid var(--success-color);
}

.notification--error {
  border-left: 4px solid var(--danger-color);
}

.notification--warning {
  border-left: 4px solid var(--warning-color);
}

.notification--info {
  border-left: 4px solid var(--primary-blue);
}

.notification-content {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
}

.notification-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
  margin-top: 2px;
}

.notification-message {
  flex: 1;
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.4;
}

.notification-close {
  background: none;
  border: none;
  color: var(--text-secondary);
  cursor: pointer;
  padding: 2px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.notification-close:hover {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.notification-close i {
  width: 16px;
  height: 16px;
}

/* Responsive Design */
@media (max-width: 768px) {
  .container {
    padding: 0 var(--space-12);
  }

  .header-content {
    flex-direction: column;
    align-items: stretch;
    gap: var(--space-12);
  }

  .header-actions {
    justify-content: center;
  }

  .brand {
    justify-content: center;
  }

  .diagnosis-result {
    grid-template-columns: 1fr;
    gap: var(--space-20);
  }

  .info-grid {
    grid-template-columns: 1fr;
  }

  .regions-grid {
    grid-template-columns: 1fr;
  }

  .tech-grid {
    grid-template-columns: 1fr;
  }

  .neuron {
    width: 8px;
    height: 8px;
  }

  .synapse {
    height: 2px;
  }

  .notification {
    left: 16px;
    right: 16px;
    max-width: none;
    transform: translateY(-100px);
  }
  
  .notification--show {
    transform: translateY(0);
  }
}

@media (max-width: 480px) {
  .main-content {
    padding: var(--space-16) 0;
  }

  .card-header,
  .card-body {
    padding: var(--space-16);
  }

  .confidence-badge {
    padding: var(--space-8);
  }

  .recommendation-item {
    padding: var(--space-12);
    gap: var(--space-12);
  }
}

/* Reduced motion preferences */
@media (prefers-reduced-motion: reduce) {
  .neuron,
  .synapse,
  .neuron::before {
    animation: none !important;
  }
  
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}

/* Print Styles */
@media print {
  .background-container,
  .header-actions {
    display: none !important;
  }

  .card {
    break-inside: avoid;
    box-shadow: none;
  }

  body {
    background: white !important;
    color: black !important;
  }
}
  </style>
</head>
<body>
  <!-- Animated Background -->
  <div class="background-container">
    <div class="neuron neuron-1"></div>
    <div class="neuron neuron-2"></div>
    <div class="neuron neuron-3"></div>
    <div class="neuron neuron-4"></div>
    <div class="neuron neuron-5"></div>
  </div>

  <!-- Header -->
  <header class="header">
    <div class="container">
      <div class="header-content">
        <button class="back-button" onclick="goBack()">
          <i data-lucide="arrow-left"></i>
          Back to Analysis
        </button>
        <div class="brand">
          <i data-lucide="brain"></i>
          <h1>JAAM AI</h1>
        </div>
        <div class="header-actions">
          <button class="btn btn--secondary" onclick="downloadReport()">
            <i data-lucide="download"></i>
            Download Report
          </button>
          <button class="btn btn--outline" onclick="printResults()">
            <i data-lucide="printer"></i>
            Print
          </button>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="main-content">
    <div class="container">
      <!-- Patient Information -->
      <section class="patient-info">
        <div class="card">
          <div class="card-header">
            <h2>Patient Scan Information</h2>
            <span class="status status--processing">Analysis Complete</span>
          </div>
          <div class="card-body">
            <div class="info-grid">
              <div class="info-item"><span class="label">Patient ID</span><span class="value">JAM-2025-0847</span></div>
              <div class="info-item"><span class="label">Scan Date</span><span class="value">October 2, 2025</span></div>
              <div class="info-item"><span class="label">Processing Time</span><span class="value">2.34 seconds</span></div>
              <div class="info-item"><span class="label">Model Used</span><span class="value">DenseNet201 (99.67% accuracy)</span></div>
            </div>
          </div>
        </div>
      </section>

      <!-- AI Results -->
      <section class="primary-diagnosis">
        <div class="card">
          <div class="card-header">
            <h2>AI Analysis Results</h2>
            <div class="confidence-badge">
  <span class="confidence-value">{{ confidence }}%</span>   <!-- was 87.3% -->
  <span class="confidence-label">Confidence</span>
</div>

            </div>
          </div>
          <div class="card-body">
            <div class="diagnosis-result">
              <div class="diagnosis-main">
                <h3>Primary Classification</h3>
                <div class="diagnosis-classification">
  <span class="classification-text">{{ label }}</span>      <!-- was "Mild Dementia (MD)" -->
  <!-- you can keep/remove risk indicator depending on if you calculate it -->
</div>

              </div>
              <div class="confidence-breakdown">
  <h4>Stage Classification Breakdown</h4>
  <div class="progress-bars">
    {% set stages = [
      ("Normal Dementia (ND)", 4.2, "#10b981"),
      ("Very Mild Dementia (VMD)", 8.5, "#f59e0b"),
      ("Mild Dementia (MD)", 87.3, "#ef4444"),
      ("Moderate Dementia (MoD)", 0.0, "#dc2626")
    ] %}
    
    {% set stage_mapping = {
      "NonDemented": "Normal Dementia (ND)",
      "VeryMildDemented": "Very Mild Dementia (VMD)",
      "MildDemented": "Mild Dementia (MD)",
      "ModerateDemented": "Moderate Dementia (MoD)"
    } %}
    
    {% set active_stage = stage_mapping.get(label) %}
    
    {% for stage_label, perc, color in stages %}
    <div class="progress-item {% if stage_label == active_stage %}active{% endif %}">
      <div class="progress-label">
        <span>{{ stage_label }}</span>
        <span>{{ perc }}%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:{{ perc }}%; background:{{ color }};"></div>
      </div>
    </div>
    {% endfor %}
  </div>
</div>

      </section>

    <!-- Brain Region Analysis -->
<section class="brain-analysis">
  <div class="card">
    <div class="card-header">
      <h2>Brain Region Analysis</h2>
      <button class="expand-button" onclick="toggleExpand(this)">
        <i data-lucide="chevron-down"></i>
      </button>
    </div>
    <div class="card-body">
      <div class="gradcam-section" style="display: flex; flex-direction: column; gap: 16px; align-items: center;">

        <!-- Original MRI -->
        {% if img_url %}
        <div class="image-container">
          <h4>Original MRI</h4>
          <img src="{{ img_url }}" alt="Original MRI" style="max-width: 500px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        </div>
        {% endif %}

        <!-- Grad-CAM Overlay -->
        {% if gradcam_url %}
        <div class="image-container">
          <h4>Grad-CAM Heatmap (VGG19)</h4>
          <img src="{{ gradcam_url }}" alt="Grad-CAM Heatmap" style="max-width: 500px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
          <p class="placeholder-text">Brain region activation patterns</p>
        </div>
        {% endif %}

      </div>
    </div>
  </div>



            <div class="regions-grid">
              <div class="region-item"><div class="region-header"><span class="region-name">Hippocampus</span><span class="severity severity--moderate">Moderate</span></div><div class="region-bar"><div class="region-fill" style="width:68%;background:#f59e0b;"></div></div><span class="region-percentage">68% affected</span></div>
              <div class="region-item"><div class="region-header"><span class="region-name">Temporal Lobe</span><span class="severity severity--mild">Mild</span></div><div class="region-bar"><div class="region-fill" style="width:34%;background:#f59e0b;"></div></div><span class="region-percentage">34% affected</span></div>
              <div class="region-item"><div class="region-header"><span class="region-name">Frontal Cortex</span><span class="severity severity--mild">Mild</span></div><div class="region-bar"><div class="region-fill" style="width:29%;background:#f59e0b;"></div></div><span class="region-percentage">29% affected</span></div>
              <div class="region-item"><div class="region-header"><span class="region-name">Parietal Lobe</span><span class="severity severity--normal">Non</span></div><div class="region-bar"><div class="region-fill" style="width:12%;background:#10b981;"></div></div><span class="region-percentage">12% affected</span></div>
            </div>
          </div>
        </div>
      </section>

      <!-- Recommendations -->
      <section class="recommendations">
        <div class="card">
          <div class="card-header">
            <h2>Clinical Recommendations</h2>
            <i data-lucide="clipboard-list"></i>
          </div>
          <div class="card-body">
            <div class="recommendations-list">
              <div class="recommendation-item priority-high"><i data-lucide="calendar-clock"></i><div class="recommendation-content"><h4>Immediate Action Required</h4><p>Schedule neurological consultation within 2 weeks</p></div></div>
              <div class="recommendation-item priority-medium"><i data-lucide="brain"></i><div class="recommendation-content"><h4>Cognitive Assessment</h4><p>Cognitive assessment battery recommended</p></div></div>
              <div class="recommendation-item priority-medium"><i data-lucide="activity"></i><div class="recommendation-content"><h4>Follow-up Imaging</h4><p>Follow-up MRI scan in 6 months</p></div></div>
              <div class="recommendation-item priority-low"><i data-lucide="heart-handshake"></i><div class="recommendation-content"><h4>Lifestyle Support</h4><p>Consider lifestyle interventions and support services</p></div></div>
            </div>
          </div>
        </div>
      </section>

      <!-- Technical Details -->
      <section class="technical-details">
        <div class="card">
          <div class="card-header">
            <h2>Technical Details</h2>
            <button class="expand-button" onclick="toggleExpand(this)"><i data-lucide="chevron-down"></i></button>
          </div>
          <div class="card-body collapsed">
            <div class="tech-grid">
              <div class="tech-item"><h4>Model Architecture</h4><p>DenseNet201 with Grad-CAM visualization</p></div>
              <div class="tech-item"><h4>Preprocessing</h4><p>Skull stripping, intensity normalization, spatial registration</p></div>
              <div class="tech-item"><h4>Data Augmentation</h4><p>Rotation, zoom, brightness adjustment during training</p></div>
              <div class="tech-item"><h4>Regularization</h4><p>L2 regularization, dropout 0.5, batch normalization</p></div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </main>

  <!-- ✅ Inline JavaScript -->
  <script>
    // Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Lucide icons
    lucide.createIcons();
    
    // Initialize animations and interactions
    initializeAnimations();
    initializeProgressBars();
    initializeInteractions();
});

// Initialize animations for elements
function initializeAnimations() {
    // Animate progress bars on load
    setTimeout(() => {
        const progressFills = document.querySelectorAll('.progress-fill');
        progressFills.forEach(fill => {
            const width = fill.style.width;
            fill.style.width = '0%';
            setTimeout(() => {
                fill.style.width = width;
            }, 100);
        });
    }, 500);

    // Animate region bars
    setTimeout(() => {
        const regionFills = document.querySelectorAll('.region-fill');
        regionFills.forEach(fill => {
            const width = fill.style.width;
            fill.style.width = '0%';
            setTimeout(() => {
                fill.style.width = width;
            }, 100);
        });
    }, 800);

    // Animate cards on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe cards for animation
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        observer.observe(card);
    });
}

// Initialize progress bar interactions
function initializeProgressBars() {
    const progressItems = document.querySelectorAll('.progress-item');
    
    progressItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            // Highlight the hovered item
            progressItems.forEach(i => i.classList.remove('highlighted'));
            this.classList.add('highlighted');
        });
        
        item.addEventListener('mouseleave', function() {
            this.classList.remove('highlighted');
        });
    });
}

// Initialize general interactions
function initializeInteractions() {
    // Add hover effects to recommendation items
    const recommendationItems = document.querySelectorAll('.recommendation-item');
    recommendationItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateX(8px)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateX(0)';
        });
    });

    // Add click effects to region items
    const regionItems = document.querySelectorAll('.region-item');
    regionItems.forEach(item => {
        item.addEventListener('click', function() {
            // Toggle selection
            regionItems.forEach(i => i.classList.remove('selected'));
            this.classList.add('selected');
            
            // Show detailed info (placeholder)
            showRegionDetails(this);
        });
    });

    // Initialize tooltips for technical terms
    initializeTooltips();
}

// Back button functionality
function goBack() {
    // Add loading state
    const backButton = document.querySelector('.back-button');
    const originalText = backButton.innerHTML;
    
    backButton.innerHTML = '<i data-lucide="loader-2"></i> Going back...';
    backButton.style.opacity = '0.7';
    backButton.disabled = true;
    
    // Reinitialize icons for the loader
    lucide.createIcons();
    
    // Simulate navigation delay
    setTimeout(() => {
        // In a real application, this would navigate to the previous page
        showNotification('Returning to analysis page...', 'info');
        
        // Reset button after delay
        setTimeout(() => {
            backButton.innerHTML = originalText;
            backButton.style.opacity = '1';
            backButton.disabled = false;
            lucide.createIcons();
        }, 1000);
    }, 500);
}

// Download report functionality
function downloadReport() {
    const downloadButton = document.querySelector('.btn--secondary');
    const originalContent = downloadButton.innerHTML;
    
    // Show loading state
    downloadButton.innerHTML = '<i data-lucide="loader-2"></i> Generating...';
    downloadButton.disabled = true;
    lucide.createIcons();
    
    // Simulate report generation
    setTimeout(() => {
        // Create a simple text report
        const reportData = generateReportData();
        downloadTextFile(reportData, 'JAAM-AI-Analysis-JAM-2025-0847.txt');
        
        // Reset button
        downloadButton.innerHTML = originalContent;
        downloadButton.disabled = false;
        lucide.createIcons();
        
        showNotification('Analysis report downloaded successfully!', 'success');
    }, 2000);
}

// Print results functionality
function printResults() {
    // Hide interactive elements for printing
    const elementsToHide = document.querySelectorAll('.header-actions, .expand-button');
    elementsToHide.forEach(el => el.style.display = 'none');
    
    // Print the page
    window.print();
    
    // Restore elements after printing
    setTimeout(() => {
        elementsToHide.forEach(el => el.style.display = '');
    }, 1000);
    
    showNotification('Preparing document for printing...', 'info');
}

// Toggle expand functionality
function toggleExpand(button) {
    const card = button.closest('.card');
    const cardBody = card.querySelector('.card-body');
    const icon = button.querySelector('i');
    
    // Toggle collapsed state
    cardBody.classList.toggle('collapsed');
    button.classList.toggle('expanded');
    
    // Update icon
    if (cardBody.classList.contains('collapsed')) {
        icon.setAttribute('data-lucide', 'chevron-down');
    } else {
        icon.setAttribute('data-lucide', 'chevron-up');
    }
    
    // Reinitialize icons
    lucide.createIcons();
    
    // Smooth animation
    if (!cardBody.classList.contains('collapsed')) {
        cardBody.style.maxHeight = cardBody.scrollHeight + 'px';
    } else {
        cardBody.style.maxHeight = '0px';
    }
}

// Generate report data
function generateReportData() {
    const currentDate = new Date().toLocaleDateString();
    
    return `JAAM AI - Analysis Report
========================

Patient Information:
- Patient ID: JAM-2025-0847
- Scan Date: October 2, 2025
- Analysis Date: ${currentDate}
- Processing Time: 2.34 seconds
- Model Used: DenseNet201 (99.67% accuracy)

Primary Diagnosis:
- Classification: Mild Dementia (MD)
- Confidence Level: 87.3%
- Risk Assessment: Moderate Risk

Stage Classification Breakdown:
- Normal Dementia (ND): 4.2%
- Very Mild Dementia (VMD): 8.5%
- Mild Dementia (MD): 87.3% [PRIMARY]
- Moderate Dementia (MoD): 0.0%

Brain Region Analysis:
- Hippocampus: 68% affected (Moderate severity)
- Temporal Lobe: 34% affected (Mild severity)
- Frontal Cortex: 29% affected (Mild severity)
- Parietal Lobe: 12% affected (Normal)

Clinical Recommendations:
1. [HIGH PRIORITY] Schedule neurological consultation within 2 weeks
2. [MEDIUM PRIORITY] Cognitive assessment battery recommended
3. [MEDIUM PRIORITY] Follow-up MRI scan in 6 months
4. [LOW PRIORITY] Consider lifestyle interventions and support services

Technical Details:
- Model Architecture: DenseNet201 with Grad-CAM visualization
- Preprocessing: Skull stripping, intensity normalization, spatial registration
- Data Augmentation: Rotation, zoom, brightness adjustment during training
- Regularization: L2 regularization, dropout 0.5, batch normalization

---
Report generated by JAAM AI Analysis System
For clinical use only - please consult with healthcare professionals
`;
}

// Download text file utility
function downloadTextFile(content, filename) {
    const element = document.createElement('a');
    const file = new Blob([content], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

// Show region details (placeholder functionality)
function showRegionDetails(regionElement) {
    const regionName = regionElement.querySelector('.region-name').textContent;
    const severity = regionElement.querySelector('.severity').textContent;
    const percentage = regionElement.querySelector('.region-percentage').textContent;
    
    showNotification(`${regionName}: ${severity} severity, ${percentage}`, 'info');
}

// Notification system
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification--${type}`;
    
    // Set icon based on type
    let icon = 'info';
    switch(type) {
        case 'success': icon = 'check-circle'; break;
        case 'error': icon = 'alert-circle'; break;
        case 'warning': icon = 'alert-triangle'; break;
        default: icon = 'info';
    }
    
    notification.innerHTML = `
        <div class="notification-content">
            <i data-lucide="${icon}" class="notification-icon"></i>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="closeNotification(this)">
                <i data-lucide="x"></i>
            </button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Initialize icons in notification
    lucide.createIcons();
    
    // Show notification with animation
    setTimeout(() => {
        notification.classList.add('notification--show');
    }, 100);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        if (document.body.contains(notification)) {
            closeNotification(notification.querySelector('.notification-close'));
        }
    }, 4000);
}

// Close notification
function closeNotification(button) {
    const notification = button.closest('.notification');
    if (notification) {
        notification.classList.remove('notification--show');
        setTimeout(() => {
            if (document.body.contains(notification)) {
                notification.remove();
            }
        }, 300);
    }
}

// Initialize tooltips for technical terms
function initializeTooltips() {
    const technicalTerms = {
        'DenseNet201': 'A deep convolutional neural network architecture with dense connections between layers',
        'Grad-CAM': 'Gradient-weighted Class Activation Mapping - shows which parts of the image influenced the AI decision',
        'Hippocampus': 'A brain region crucial for memory formation, often affected early in dementia',
        'Temporal Lobe': 'Brain region involved in processing auditory information and language',
        'Frontal Cortex': 'Brain region responsible for executive functions and decision making',
        'Parietal Lobe': 'Brain region involved in spatial processing and attention'
    };
    
    // Add tooltip functionality to technical terms
    Object.keys(technicalTerms).forEach(term => {
        const elements = document.querySelectorAll(`*:contains("${term}")`);
        elements.forEach(element => {
            if (element.children.length === 0) { // Only text nodes
                element.title = technicalTerms[term];
                element.style.cursor = 'help';
                element.style.borderBottom = '1px dotted var(--text-secondary)';
            }
        });
    });
}

// Utility function to find elements containing text
function findElementsContaining(text) {
    const elements = [];
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    
    let node;
    while (node = walker.nextNode()) {
        if (node.textContent.includes(text)) {
            elements.push(node.parentElement);
        }
    }
    
    return elements;
}

// Handle keyboard navigation
document.addEventListener('keydown', function(e) {
    // Close notifications with Escape
    if (e.key === 'Escape') {
        const notification = document.querySelector('.notification');
        if (notification) {
            closeNotification(notification.querySelector('.notification-close'));
        }
    }
    
    // Print with Ctrl+P
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        printResults();
    }
    
    // Download with Ctrl+D
    if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        downloadReport();
    }
});

// Handle window resize for responsive behavior
window.addEventListener('resize', function() {
    // Adjust neuron animations for smaller screens
    if (window.innerWidth <= 768) {
        const neurons = document.querySelectorAll('.neuron');
        neurons.forEach(neuron => {
            neuron.style.width = '6px';
            neuron.style.height = '6px';
        });
        
        const synapses = document.querySelectorAll('.synapse');
        synapses.forEach(synapse => {
            synapse.style.height = '1px';
        });
    } else {
        const neurons = document.querySelectorAll('.neuron');
        neurons.forEach(neuron => {
            neuron.style.width = '8px';
            neuron.style.height = '8px';
        });
        
        const synapses = document.querySelectorAll('.synapse');
        synapses.forEach(synapse => {
            synapse.style.height = '2px';
        });
    }
});

// Performance optimization: Reduce animations on low-end devices
function optimizeForPerformance() {
    // Check if device prefers reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const animatedElements = document.querySelectorAll('.neuron, .synapse');
        animatedElements.forEach(element => {
            element.style.animation = 'none';
        });
        
        // Reduce transition durations
        document.documentElement.style.setProperty('--animation-duration', '0.1s');
    }
}

// Initialize performance optimizations
document.addEventListener('DOMContentLoaded', optimizeForPerformance);

// Add smooth scrolling for any internal links
document.addEventListener('DOMContentLoaded', function() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Initialize service worker for offline functionality (if needed)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        // Register service worker if available
        // This is optional for the demo
    });
}

// Add loading states to interactive elements
function addLoadingState(element, text = 'Loading...') {
    const originalContent = element.innerHTML;
    element.style.opacity = '0.7';
    element.disabled = true;
    element.innerHTML = `<i data-lucide="loader-2"></i> ${text}`;
    lucide.createIcons();
    
    return function removeLoadingState() {
        element.style.opacity = '1';
        element.disabled = false;
        element.innerHTML = originalContent;
        lucide.createIcons();
    };
}

// Add CSS for additional interactive states
const additionalStyles = `
.region-item.selected {
    border-color: var(--primary-blue);
    background: rgba(59, 130, 246, 0.1);
}

.progress-item.highlighted {
    opacity: 1;
    transform: scale(1.02);
}

.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    max-width: 400px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-base);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
    transform: translateX(calc(100% + 20px));
    transition: transform 0.3s ease;
}

.notification--show {
    transform: translateX(0);
}

.notification--success {
    border-left: 4px solid var(--success-color);
}

.notification--error {
    border-left: 4px solid var(--danger-color);
}

.notification--warning {
    border-left: 4px solid var(--warning-color);
}

.notification--info {
    border-left: 4px solid var(--primary-blue);
}

.notification-content {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 16px;
}

.notification-icon {
    width: 20px;
    height: 20px;
    flex-shrink: 0;
    margin-top: 2px;
}

.notification-message {
    flex: 1;
    color: var(--text-primary);
    font-size: 14px;
    line-height: 1.4;
}

.notification-close {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 2px;
    border-radius: 4px;
    transition: all 0.2s ease;
}

.notification-close:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
}

.notification-close i {
    width: 16px;
    height: 16px;
}

@media (max-width: 768px) {
    .notification {
        left: 16px;
        right: 16px;
        max-width: none;
        transform: translateY(-100px);
    }
    
    .notification--show {
        transform: translateY(0);
    }
}

.card {
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.6s ease, transform 0.6s ease;
}

.card.animate-in {
    opacity: 1;
    transform: translateY(0);
}

@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
`;

// Add additional styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

// Export functions for testing (if in a module environment)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        goBack,
        downloadReport,
        printResults,
        toggleExpand,
        showNotification,
        closeNotification
    };
}
  </script>
</body>
</html>
'''

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def home(): 
    return render_template_string(UPLOAD_HTML)

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    label, conf, gradcam_path = predict_ensemble(path)

    result_path = os.path.join(RESULTS_FOLDER,"result_"+f.filename)
    Image.open(path).save(result_path)

    img_url = url_for('results_file',filename="result_"+f.filename)
    gradcam_url = url_for('results_file',filename=os.path.basename(gradcam_path)) if gradcam_path else None

    return render_template_string(
        RESULT_HTML,
        label=label,
        confidence=round(conf,2),
        img_url=img_url,
        gradcam_url=gradcam_url
    )

@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(RESULTS_FOLDER,filename)
  
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
      
        logging.info("Received prediction request")
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        label, confidence, _ = predict_ensemble(filepath)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500  

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"Running Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)


