/**
 * Atlas Premium Design System
 *
 * Enterprise-grade design tokens, components, and theming.
 * Consistent visual language across all Atlas interfaces.
 */

export const ATLAS_DESIGN_SYSTEM = {
  // ===========================================
  // COLOR PALETTE
  // ===========================================
  colors: {
    // Brand colors
    brand: {
      primary: '#6366F1',      // Indigo
      secondary: '#8B5CF6',    // Violet
      accent: '#06B6D4',       // Cyan
    },

    // Semantic colors
    semantic: {
      success: '#10B981',      // Emerald
      warning: '#F59E0B',      // Amber
      error: '#EF4444',        // Red
      info: '#3B82F6',         // Blue
    },

    // Chart colors (colorblind-friendly palette)
    chart: [
      '#6366F1',  // Indigo
      '#10B981',  // Emerald
      '#F59E0B',  // Amber
      '#EF4444',  // Red
      '#8B5CF6',  // Violet
      '#06B6D4',  // Cyan
      '#EC4899',  // Pink
      '#14B8A6',  // Teal
    ],

    // Gradient presets
    gradients: {
      primary: 'linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)',
      success: 'linear-gradient(135deg, #10B981 0%, #14B8A6 100%)',
      warning: 'linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%)',
      error: 'linear-gradient(135deg, #EF4444 0%, #F87171 100%)',
      dark: 'linear-gradient(135deg, #1F2937 0%, #111827 100%)',
    },
  },

  // ===========================================
  // TYPOGRAPHY
  // ===========================================
  typography: {
    fontFamily: {
      sans: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif',
      mono: '"JetBrains Mono", "Fira Code", Consolas, Monaco, monospace',
    },

    fontSize: {
      xs: '10px',
      sm: '12px',
      base: '14px',
      lg: '16px',
      xl: '18px',
      '2xl': '24px',
      '3xl': '32px',
    },

    fontWeight: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },

    lineHeight: {
      tight: 1.25,
      normal: 1.5,
      relaxed: 1.75,
    },
  },

  // ===========================================
  // SPACING
  // ===========================================
  spacing: {
    px: '1px',
    0: '0',
    1: '4px',
    2: '8px',
    3: '12px',
    4: '16px',
    5: '20px',
    6: '24px',
    8: '32px',
    10: '40px',
    12: '48px',
    16: '64px',
  },

  // ===========================================
  // BORDERS & RADIUS
  // ===========================================
  borders: {
    radius: {
      none: '0',
      sm: '4px',
      md: '8px',
      lg: '12px',
      xl: '16px',
      full: '9999px',
    },

    width: {
      none: '0',
      thin: '1px',
      medium: '2px',
      thick: '4px',
    },
  },

  // ===========================================
  // SHADOWS
  // ===========================================
  shadows: {
    none: 'none',
    sm: '0 1px 2px rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
    inner: 'inset 0 2px 4px rgba(0, 0, 0, 0.05)',
    glow: '0 0 20px rgba(99, 102, 241, 0.3)',
  },

  // ===========================================
  // TRANSITIONS
  // ===========================================
  transitions: {
    duration: {
      fast: '100ms',
      normal: '200ms',
      slow: '300ms',
    },

    easing: {
      linear: 'linear',
      ease: 'ease',
      easeIn: 'ease-in',
      easeOut: 'ease-out',
      easeInOut: 'ease-in-out',
      spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
    },
  },

  // ===========================================
  // Z-INDEX SCALE
  // ===========================================
  zIndex: {
    base: 0,
    dropdown: 100,
    sticky: 200,
    fixed: 300,
    overlay: 400,
    modal: 500,
    popover: 600,
    tooltip: 700,
  },

  // ===========================================
  // BREAKPOINTS
  // ===========================================
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px',
  },
};

/**
 * Generate CSS custom properties from design tokens
 */
export function generateCSSVariables(): string {
  const { colors, typography, spacing, borders, shadows, transitions } = ATLAS_DESIGN_SYSTEM;

  return `
    :root {
      /* Brand Colors */
      --atlas-brand-primary: ${colors.brand.primary};
      --atlas-brand-secondary: ${colors.brand.secondary};
      --atlas-brand-accent: ${colors.brand.accent};

      /* Semantic Colors */
      --atlas-success: ${colors.semantic.success};
      --atlas-warning: ${colors.semantic.warning};
      --atlas-error: ${colors.semantic.error};
      --atlas-info: ${colors.semantic.info};

      /* Typography */
      --atlas-font-sans: ${typography.fontFamily.sans};
      --atlas-font-mono: ${typography.fontFamily.mono};

      /* Font Sizes */
      --atlas-text-xs: ${typography.fontSize.xs};
      --atlas-text-sm: ${typography.fontSize.sm};
      --atlas-text-base: ${typography.fontSize.base};
      --atlas-text-lg: ${typography.fontSize.lg};
      --atlas-text-xl: ${typography.fontSize.xl};
      --atlas-text-2xl: ${typography.fontSize['2xl']};
      --atlas-text-3xl: ${typography.fontSize['3xl']};

      /* Spacing */
      --atlas-space-1: ${spacing[1]};
      --atlas-space-2: ${spacing[2]};
      --atlas-space-3: ${spacing[3]};
      --atlas-space-4: ${spacing[4]};
      --atlas-space-6: ${spacing[6]};
      --atlas-space-8: ${spacing[8]};

      /* Border Radius */
      --atlas-radius-sm: ${borders.radius.sm};
      --atlas-radius-md: ${borders.radius.md};
      --atlas-radius-lg: ${borders.radius.lg};
      --atlas-radius-xl: ${borders.radius.xl};
      --atlas-radius-full: ${borders.radius.full};

      /* Shadows */
      --atlas-shadow-sm: ${shadows.sm};
      --atlas-shadow-md: ${shadows.md};
      --atlas-shadow-lg: ${shadows.lg};
      --atlas-shadow-glow: ${shadows.glow};

      /* Transitions */
      --atlas-transition-fast: ${transitions.duration.fast};
      --atlas-transition-normal: ${transitions.duration.normal};
      --atlas-transition-slow: ${transitions.duration.slow};
    }
  `;
}

/**
 * Premium component styles
 */
export const COMPONENT_STYLES = {
  // Card component
  card: `
    .atlas-card {
      background: var(--vscode-editor-background);
      border: 1px solid var(--vscode-panel-border);
      border-radius: var(--atlas-radius-lg);
      padding: var(--atlas-space-4);
      transition: all var(--atlas-transition-normal) ease;
    }

    .atlas-card:hover {
      border-color: var(--atlas-brand-primary);
      box-shadow: var(--atlas-shadow-glow);
    }

    .atlas-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: var(--atlas-space-3);
    }

    .atlas-card-title {
      font-size: var(--atlas-text-lg);
      font-weight: 600;
      color: var(--vscode-editor-foreground);
    }
  `,

  // Button component
  button: `
    .atlas-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: var(--atlas-space-2);
      padding: var(--atlas-space-2) var(--atlas-space-4);
      font-size: var(--atlas-text-sm);
      font-weight: 500;
      border-radius: var(--atlas-radius-md);
      border: none;
      cursor: pointer;
      transition: all var(--atlas-transition-fast) ease;
    }

    .atlas-btn-primary {
      background: var(--atlas-brand-primary);
      color: white;
    }

    .atlas-btn-primary:hover {
      background: var(--atlas-brand-secondary);
      transform: translateY(-1px);
    }

    .atlas-btn-secondary {
      background: transparent;
      border: 1px solid var(--vscode-panel-border);
      color: var(--vscode-editor-foreground);
    }

    .atlas-btn-secondary:hover {
      border-color: var(--atlas-brand-primary);
      color: var(--atlas-brand-primary);
    }

    .atlas-btn-ghost {
      background: transparent;
      color: var(--vscode-descriptionForeground);
    }

    .atlas-btn-ghost:hover {
      background: var(--vscode-list-hoverBackground);
      color: var(--vscode-editor-foreground);
    }

    .atlas-btn-icon {
      padding: var(--atlas-space-2);
      border-radius: var(--atlas-radius-md);
    }
  `,

  // Badge component
  badge: `
    .atlas-badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      font-size: var(--atlas-text-xs);
      font-weight: 500;
      border-radius: var(--atlas-radius-full);
    }

    .atlas-badge-success {
      background: rgba(16, 185, 129, 0.15);
      color: var(--atlas-success);
    }

    .atlas-badge-warning {
      background: rgba(245, 158, 11, 0.15);
      color: var(--atlas-warning);
    }

    .atlas-badge-error {
      background: rgba(239, 68, 68, 0.15);
      color: var(--atlas-error);
    }

    .atlas-badge-info {
      background: rgba(59, 130, 246, 0.15);
      color: var(--atlas-info);
    }
  `,

  // Metric display component
  metric: `
    .atlas-metric {
      display: flex;
      flex-direction: column;
      padding: var(--atlas-space-3);
      background: var(--vscode-sideBar-background);
      border-radius: var(--atlas-radius-md);
    }

    .atlas-metric-label {
      font-size: var(--atlas-text-xs);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--vscode-descriptionForeground);
      margin-bottom: var(--atlas-space-1);
    }

    .atlas-metric-value {
      font-size: var(--atlas-text-2xl);
      font-weight: 700;
      font-family: var(--atlas-font-mono);
      color: var(--vscode-editor-foreground);
    }

    .atlas-metric-trend {
      display: flex;
      align-items: center;
      gap: 4px;
      font-size: var(--atlas-text-xs);
      margin-top: var(--atlas-space-1);
    }

    .atlas-metric-trend-up { color: var(--atlas-error); }
    .atlas-metric-trend-down { color: var(--atlas-success); }
    .atlas-metric-trend-flat { color: var(--vscode-descriptionForeground); }
  `,

  // Insight card component
  insight: `
    .atlas-insight {
      display: flex;
      gap: var(--atlas-space-3);
      padding: var(--atlas-space-3);
      background: var(--vscode-editor-background);
      border: 1px solid var(--vscode-panel-border);
      border-radius: var(--atlas-radius-md);
      transition: all var(--atlas-transition-normal) ease;
    }

    .atlas-insight:hover {
      border-color: var(--atlas-brand-primary);
    }

    .atlas-insight-icon {
      width: 32px;
      height: 32px;
      border-radius: var(--atlas-radius-md);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }

    .atlas-insight-icon-info { background: rgba(59, 130, 246, 0.15); color: var(--atlas-info); }
    .atlas-insight-icon-warning { background: rgba(245, 158, 11, 0.15); color: var(--atlas-warning); }
    .atlas-insight-icon-critical { background: rgba(239, 68, 68, 0.15); color: var(--atlas-error); }

    .atlas-insight-content { flex: 1; }

    .atlas-insight-title {
      font-weight: 600;
      color: var(--vscode-editor-foreground);
      margin-bottom: 4px;
    }

    .atlas-insight-description {
      font-size: var(--atlas-text-sm);
      color: var(--vscode-descriptionForeground);
      line-height: 1.5;
    }

    .atlas-insight-actions {
      display: flex;
      gap: var(--atlas-space-2);
      margin-top: var(--atlas-space-2);
    }
  `,

  // Progress indicator
  progress: `
    .atlas-progress {
      height: 4px;
      background: var(--vscode-panel-border);
      border-radius: var(--atlas-radius-full);
      overflow: hidden;
    }

    .atlas-progress-bar {
      height: 100%;
      background: var(--atlas-brand-primary);
      border-radius: var(--atlas-radius-full);
      transition: width var(--atlas-transition-normal) ease;
    }

    .atlas-progress-bar-animated {
      background: linear-gradient(
        90deg,
        var(--atlas-brand-primary) 0%,
        var(--atlas-brand-secondary) 50%,
        var(--atlas-brand-primary) 100%
      );
      background-size: 200% 100%;
      animation: progress-shimmer 1.5s infinite;
    }

    @keyframes progress-shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
  `,

  // Skeleton loading
  skeleton: `
    .atlas-skeleton {
      background: linear-gradient(
        90deg,
        var(--vscode-panel-border) 25%,
        var(--vscode-sideBar-background) 50%,
        var(--vscode-panel-border) 75%
      );
      background-size: 200% 100%;
      animation: skeleton-shimmer 1.5s infinite;
      border-radius: var(--atlas-radius-md);
    }

    @keyframes skeleton-shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
  `,
};

/**
 * Combine all styles into a single stylesheet
 */
export function getFullStylesheet(): string {
  return `
    ${generateCSSVariables()}
    ${Object.values(COMPONENT_STYLES).join('\n')}
  `;
}
